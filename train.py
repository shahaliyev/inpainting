import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.build import build_dataloader
from models.build import build_model
from training.checkpoint import load_checkpoint, make_checkpoint_dict, save_best_checkpoint, save_last_checkpoint
from training.engine import evaluate, train_one_epoch
from training.logger import MetricsLogger
from training.optim import build_optimizer, build_scheduler
from utils.run_metadata import build_train_run_name, save_resolved_config, save_run_metadata

CONFIGS_DIR = Path("configs")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Dataset config key, e.g. carpet")
    ap.add_argument("--mask", default=None, help="Mask config key, e.g. mixed")
    ap.add_argument("--model", default=None, help="Model config key, e.g. unet")
    ap.add_argument("--loader", default="default", help="Loader config key")
    ap.add_argument("--train", dest="train_name", default="default", help="Train config key")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_ckpt", default=None)
    return ap.parse_args()


def apply_runtime_settings(args, train_cfg):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        if bool(getattr(train_cfg, "tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


def get_amp_dtype(train_cfg, device):
    if device.type != "cuda":
        return None
    name = str(getattr(train_cfg, "amp_dtype", "bfloat16")).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported train.amp_dtype: {name}. Use bfloat16 or float16.")


def compile_model(model, train_cfg):
    compile_cfg = getattr(train_cfg, "compile", None)
    enabled = bool(getattr(compile_cfg, "enabled", False)) if compile_cfg is not None else False
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this torch build; skipping compile")
        return model
    mode = str(getattr(compile_cfg, "mode", "default"))
    fullgraph = bool(getattr(compile_cfg, "fullgraph", False))
    dynamic = bool(getattr(compile_cfg, "dynamic", False))
    return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

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


def resolve_config_path(group: str, name: str) -> str:
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown {group} config '{name}': {path}")
    return str(path)


def load_configs(config_paths):
    dataset_cfg = OmegaConf.load(config_paths["dataset_yaml"])
    loader_cfg = OmegaConf.load(config_paths["loader_yaml"])
    mask_cfg = OmegaConf.load(config_paths["mask_yaml"])
    model_cfg = OmegaConf.load(config_paths["model_yaml"])
    train_cfg = OmegaConf.load(config_paths["train_yaml"])

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
    if args.resume:
        if args.resume_ckpt is None:
            raise ValueError("--resume requires --resume_ckpt pointing to an existing checkpoint.")
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_ckpt not found: {ckpt_path}")
        if ckpt_path.parent.name != "checkpoints":
            raise ValueError(f"--resume_ckpt must be inside a checkpoints directory: {ckpt_path}")
    else:
        if not args.dataset or not args.mask or not args.model:
            raise ValueError("Fresh training requires --dataset, --mask, and --model config keys.")
        ckpt_path = None

    if args.resume:
        ckpt_raw = torch.load(Path(args.resume_ckpt), map_location="cpu")
        config_paths = dict(ckpt_raw.get("config_paths", {}) or {})
        required = {"dataset_yaml", "loader_yaml", "mask_yaml", "model_yaml", "train_yaml"}
        if not required.issubset(config_paths.keys()):
            missing = sorted(required - set(config_paths.keys()))
            raise ValueError(f"Resume checkpoint is missing config paths: {missing}")
    else:
        config_paths = {
            "dataset_yaml": resolve_config_path("dataset", args.dataset),
            "loader_yaml": resolve_config_path("loader", args.loader),
            "mask_yaml": resolve_config_path("mask", args.mask),
            "model_yaml": resolve_config_path("model", args.model),
            "train_yaml": resolve_config_path("train", args.train_name),
        }

    dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg = load_configs(config_paths)
    apply_runtime_settings(args, train_cfg)
    apply_thread_settings(train_cfg)
    apply_overrides(args, dataset_cfg, loader_cfg, train_cfg)

    epochs = int(train_cfg.epochs)
    grad_accum_steps = int(train_cfg.grad_accum_steps)
    use_amp = bool(train_cfg.mixed_precision)
    log_every = int(train_cfg.log_every_steps)
    vis_every = int(train_cfg.vis_every_steps)
    eval_every_epochs = int(train_cfg.eval_every_epochs)
    val_vis_every_epochs = int(train_cfg.val_vis_every_epochs)
    save_last_every_epochs = int(train_cfg.ckpt.save_last_every_epochs)
    save_best = bool(train_cfg.ckpt.save_best)
    patience = int(train_cfg.ckpt.patience)
    metric_scope = str(train_cfg.metrics.scope).lower()
    report_both_metrics = bool(train_cfg.metrics.report_both)
    if metric_scope not in {"mask", "full"}:
        raise ValueError(f"Unsupported train.metrics.scope: {metric_scope}. Use 'mask' or 'full'.")

    dl_train, dl_val = build_dataloaders(args, dataset_cfg, loader_cfg, mask_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(train_cfg, device)
    model_base = build_model(model_cfg).to(device)
    optimizer = build_optimizer(model_base, train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, epochs)
    use_grad_scaler = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    loss_fn = nn.L1Loss(reduction="none")

    if args.resume:
        checkpoint_dir = ckpt_path.parent
        run_dir = checkpoint_dir.parent
        run_name = run_dir.name
    else:
        run_name = build_train_run_name(
            model_yaml=config_paths["model_yaml"],
            dataset_yaml=config_paths["dataset_yaml"],
            mask_cfg=mask_cfg,
            seed=args.seed,
        )
        run_dir = Path(args.runs_dir) / run_name
        checkpoint_dir = run_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir)
    resolved_cfg = {
        "dataset_cfg": OmegaConf.to_container(dataset_cfg, resolve=True),
        "loader_cfg": OmegaConf.to_container(loader_cfg, resolve=True),
        "mask_cfg": OmegaConf.to_container(mask_cfg, resolve=True),
        "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
    }
    if not args.resume:
        save_run_metadata(
            run_dir,
            run_name=run_name,
            seed=args.seed,
            args_dict=vars(args),
            config_paths=config_paths,
            resolved_cfg=resolved_cfg,
        )
        save_resolved_config(run_dir, resolved_cfg)
    print(f"run_dir={run_dir}")
    print(
        f"startup: dataset={Path(config_paths['dataset_yaml']).stem} "
        f"mask={Path(config_paths['mask_yaml']).stem} "
        f"model={Path(config_paths['model_yaml']).stem} "
        f"train={Path(config_paths['train_yaml']).stem} "
        f"metric_scope={metric_scope}"
    )

    mean_list = list(dataset_cfg.norm.mean)
    std_list = list(dataset_cfg.norm.std)
    mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(args.resume_ckpt) if args.resume_ckpt is not None else checkpoint_dir / "last.pt"

    if args.resume:
        state = load_checkpoint(path=ckpt_path, model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, device=device)
        start_epoch = int(state["epoch"]) + 1
        step = int(state["step"])
        best_val_loss = float(state["best_val_loss"])
        raw_paths = state["raw"].get("config_paths", {}) or {}
        for key, cli_val in (("dataset_yaml", args.dataset), ("mask_yaml", args.mask), ("model_yaml", args.model)):
            if cli_val:
                ckpt_stem = Path(raw_paths.get(key, "")).stem
                if ckpt_stem and ckpt_stem != cli_val:
                    print(f"WARNING: resume checkpoint {key}='{ckpt_stem}' but CLI provided '{cli_val}'. Using checkpoint config.")
        print(f"resumed from {ckpt_path} | epoch={start_epoch} step={step} best_val_loss={best_val_loss:.6f}")

    model = compile_model(model_base, train_cfg)

    val_loss = None

    for epoch in range(start_epoch, epochs + 1):
        train_out = train_one_epoch(model=model, dl_train=dl_train, optimizer=optimizer, scaler=scaler, device=device, loss_fn=loss_fn, use_amp=use_amp, amp_dtype=amp_dtype, grad_accum_steps=grad_accum_steps, log_every=log_every, vis_every=vis_every, epoch=epoch, global_step=step, logger=logger, mean=mean, std=std)
        step = int(train_out["global_step"])
        train_loss = float(train_out["train_loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch, step=step, split="train_epoch", loss=train_loss, lr=lr_now)
        print(f"epoch={epoch} train_loss={train_loss:.6f}")

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            save_val_vis = val_vis_every_epochs > 0 and epoch % val_vis_every_epochs == 0
            val_out = evaluate(
                model=model,
                dl=dl_val,
                device=device,
                loss_fn=loss_fn,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                epoch=epoch,
                global_step=step,
                logger=logger,
                mean=mean,
                std=std,
                save_vis=save_val_vis,
                metric_scope=metric_scope,
                report_both=report_both_metrics,
            )
            val_loss = float(val_out["val_loss"])
            logger.log(epoch=epoch, step=step, split="val", loss=val_loss, lr=lr_now)
            print(f"epoch={epoch} val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    best_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
                    best_path = save_best_checkpoint(checkpoint_dir, best_ckpt)
                    print(f"saved best checkpoint: {best_path}")
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f"early stopping: no improvement for {patience} epochs (best val_loss={best_val_loss:.6f})")
                    break

        if scheduler is not None:
            scheduler.step()

        if save_last_every_epochs > 0 and epoch % save_last_every_epochs == 0:
            last_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
            last_path = save_last_checkpoint(checkpoint_dir, last_ckpt)
            print(f"saved last checkpoint: {last_path}")

    final_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=epochs, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
    final_path = save_last_checkpoint(checkpoint_dir, final_ckpt)
    print(f"done. saved: {final_path}")


if __name__ == "__main__":
    main()