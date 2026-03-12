import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.build import build_dataloader
from models.unet import build_unet
from training.checkpoint import load_checkpoint, make_checkpoint_dict, save_best_checkpoint, save_last_checkpoint
from training.engine import evaluate, train_one_epoch
from training.logger import MetricsLogger
from training.optim import build_optimizer, build_scheduler


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", default="configs/dataset/carpet.yaml")
    ap.add_argument("--loader_yaml", default="configs/loader/default.yaml")
    ap.add_argument("--mask_yaml", default="configs/mask/block.yaml")
    ap.add_argument("--model_yaml", default="configs/model/unet.yaml")
    ap.add_argument("--train_yaml", default="configs/train/default.yaml")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--exp", default="train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--eval_every_epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_threads", type=int, default=2)
    ap.add_argument("--interop_threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_ckpt", default=None)
    return ap.parse_args()


def apply_runtime_settings(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

def apply_thread_settings(train_cfg):
    threads_cfg = getattr(train_cfg, "threads", None)
    if threads_cfg is None:
        return
    if getattr(threads_cfg, "omp", None) is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads_cfg.omp)
    if getattr(threads_cfg, "mkl", None) is not None:
        os.environ["MKL_NUM_THREADS"] = str(threads_cfg.mkl)
    if getattr(threads_cfg, "torch", None) is not None:
        torch.set_num_threads(int(threads_cfg.torch))
    if getattr(threads_cfg, "interop", None) is not None:
        torch.set_num_interop_threads(int(threads_cfg.interop))


def apply_overrides(args, dataset_cfg, loader_cfg, train_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = True
        dataset_cfg.limit_seed = int(args.seed)
    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)
    if args.epochs is not None:
        train_cfg.epochs = int(args.epochs)
    if args.lr is not None:
        train_cfg.optimizer.lr = float(args.lr)
    if args.log_every is not None:
        train_cfg.log_every_steps = int(args.log_every)
    if args.eval_every_epochs is not None:
        train_cfg.eval_every_epochs = int(args.eval_every_epochs)


def load_configs(args):
    dataset_cfg = OmegaConf.load(args.dataset_yaml)
    loader_cfg = OmegaConf.load(args.loader_yaml)
    mask_cfg = OmegaConf.load(args.mask_yaml)
    model_cfg = OmegaConf.load(args.model_yaml)
    train_cfg = OmegaConf.load(args.train_yaml)

    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError("dataset_cfg.root is missing")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)

    return dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg


def build_dataloaders(args, dataset_cfg, loader_cfg, mask_cfg):
    dl_train = build_dataloader(dataset_cfg, loader_cfg, split=args.split, mask_cfg=mask_cfg, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)
    dl_val = build_dataloader(dataset_cfg, loader_cfg, split=args.val_split, mask_cfg=mask_cfg, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)
    return dl_train, dl_val


def main():
    args = parse_args()
    apply_runtime_settings(args)

    dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg = load_configs(args)
    apply_thread_settings(train_cfg)
    apply_overrides(args, dataset_cfg, loader_cfg, train_cfg)

    epochs = int(getattr(train_cfg, "epochs", 100))
    grad_accum_steps = int(getattr(train_cfg, "grad_accum_steps", 1))
    use_amp = bool(getattr(train_cfg, "mixed_precision", False))
    log_every = int(getattr(train_cfg, "log_every_steps", 100))
    vis_every = int(getattr(train_cfg, "vis_every_steps", 0))
    eval_every_epochs = int(getattr(train_cfg, "eval_every_epochs", 1))
    val_vis_every_epochs = int(getattr(train_cfg, "val_vis_every_epochs", 1))
    save_last_every_epochs = int(getattr(train_cfg.ckpt, "save_last_every_epochs", 1))
    save_best = bool(getattr(train_cfg.ckpt, "save_best", True))

    dl_train, dl_val = build_dataloaders(args, dataset_cfg, loader_cfg, mask_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    loss_fn = nn.L1Loss(reduction="none")

    run_dir = Path(args.runs_dir) / args.exp
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir)

    mean_list = list(getattr(dataset_cfg.norm, "mean", [0.5, 0.5, 0.5]))
    std_list = list(getattr(dataset_cfg.norm, "std", [0.5, 0.5, 0.5]))
    mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    ckpt_path = Path(args.resume_ckpt) if args.resume_ckpt is not None else checkpoint_dir / "last.pt"

    if args.resume:
        state = load_checkpoint(path=ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler if (use_amp and device.type == "cuda") else None, device=device)
        start_epoch = int(state["epoch"]) + 1
        step = int(state["step"])
        best_val_loss = float(state["best_val_loss"])
        print(f"resumed from {ckpt_path} | epoch={start_epoch} step={step} best_val_loss={best_val_loss:.6f}")

    val_loss = None

    for epoch in range(start_epoch, epochs + 1):
        train_out = train_one_epoch(model=model, dl_train=dl_train, optimizer=optimizer, scaler=scaler, device=device, loss_fn=loss_fn, use_amp=use_amp, grad_accum_steps=grad_accum_steps, log_every=log_every, vis_every=vis_every, epoch=epoch, global_step=step, logger=logger, mean=mean, std=std)
        step = int(train_out["global_step"])
        train_loss = float(train_out["train_loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch, step=step, split="train_epoch", loss=train_loss, lr=lr_now)
        print(f"epoch={epoch} train_loss={train_loss:.6f}")

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            save_val_vis = val_vis_every_epochs > 0 and epoch % val_vis_every_epochs == 0
            val_out = evaluate(model=model, dl=dl_val, device=device, loss_fn=loss_fn, use_amp=use_amp, epoch=epoch, global_step=step, logger=logger, mean=mean, std=std, save_vis=save_val_vis)
            val_loss = float(val_out["val_loss"])
            logger.log(epoch=epoch, step=step, split="val", loss=val_loss, lr=lr_now)
            print(f"epoch={epoch} val_loss={val_loss:.6f}")

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = make_checkpoint_dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler if (use_amp and device.type == "cuda") else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
                best_path = save_best_checkpoint(checkpoint_dir, best_ckpt)
                print(f"saved best checkpoint: {best_path}")

        if scheduler is not None:
            scheduler.step()

        if save_last_every_epochs > 0 and epoch % save_last_every_epochs == 0:
            last_ckpt = make_checkpoint_dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler if (use_amp and device.type == "cuda") else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
            last_path = save_last_checkpoint(checkpoint_dir, last_ckpt)
            print(f"saved last checkpoint: {last_path}")

    final_ckpt = make_checkpoint_dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler if (use_amp and device.type == "cuda") else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, epoch=epochs, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
    final_path = save_last_checkpoint(checkpoint_dir, final_ckpt)
    print(f"done. saved: {final_path}")


if __name__ == "__main__":
    main()