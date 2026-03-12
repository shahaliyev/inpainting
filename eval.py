import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.build import build_dataloader
from models.unet import build_unet
from training.checkpoint import load_checkpoint
from training.engine import evaluate
from training.logger import MetricsLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", default="configs/dataset/carpet.yaml")
    ap.add_argument("--loader_yaml", default="configs/loader/default.yaml")
    ap.add_argument("--mask_yaml", default="configs/mask/block.yaml")
    ap.add_argument("--model_yaml", default="configs/model/unet.yaml")
    ap.add_argument("--train_yaml", default="configs/train/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--exp", default="eval")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_vis", action="store_true")
    return ap.parse_args()


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


def apply_overrides(args, dataset_cfg, loader_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = False
        dataset_cfg.limit_seed = int(args.seed)

    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg = load_configs(args)
    apply_overrides(args, dataset_cfg, loader_cfg)

    use_amp = bool(getattr(train_cfg, "mixed_precision", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = build_dataloader(dataset_cfg, loader_cfg, split=args.split, mask_cfg=mask_cfg, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)

    model = build_unet(model_cfg).to(device)
    loss_fn = nn.L1Loss(reduction="none")

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    state = load_checkpoint(path=args.ckpt, model=model, optimizer=None, scheduler=None, scaler=scaler if (use_amp and device.type == "cuda") else None, device=device)

    run_dir = Path(args.runs_dir) / args.exp
    logger = MetricsLogger(run_dir) if args.save_vis else None

    mean_list = list(getattr(dataset_cfg.norm, "mean", [0.5, 0.5, 0.5]))
    std_list = list(getattr(dataset_cfg.norm, "std", [0.5, 0.5, 0.5]))
    mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    out = evaluate(model=model, dl=dl, device=device, loss_fn=loss_fn, use_amp=use_amp, epoch=state["epoch"], global_step=state["step"], logger=logger, mean=mean, std=std, save_vis=args.save_vis)
    val_loss = float(out["val_loss"])

    print(f"checkpoint={args.ckpt}")
    print(f"epoch={state['epoch']} step={state['step']}")
    print(f"split={args.split} loss={val_loss:.6f}")

    if logger is not None:
        logger.log(epoch=state["epoch"], step=state["step"], split=args.split, loss=val_loss, lr=0.0)


if __name__ == "__main__":
    main()