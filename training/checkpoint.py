from pathlib import Path

import torch
from omegaconf import OmegaConf

CKPT_FORMAT_VERSION = 1


def validate_checkpoint_schema(ckpt: dict):
    ver = int(ckpt.get("ckpt_format_version", 0))
    if ver != CKPT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {ver}. "
            f"Expected {CKPT_FORMAT_VERSION}. Re-train with current code."
        )


def make_checkpoint_dict(
    model,
    optimizer,
    scheduler,
    scaler,
    model_cfg,
    dataset_cfg,
    loader_cfg,
    mask_cfg,
    train_cfg,
    config_paths,
    epoch,
    step,
    seed,
    best_val_loss,
    last_val_loss=None,
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        "dataset_cfg": OmegaConf.to_container(dataset_cfg, resolve=True),
        "loader_cfg": OmegaConf.to_container(loader_cfg, resolve=True),
        "mask_cfg": OmegaConf.to_container(mask_cfg, resolve=True),
        "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
        "config_paths": dict(config_paths),
        "ckpt_format_version": CKPT_FORMAT_VERSION,
        "epoch": int(epoch),
        "step": int(step),
        "seed": int(seed),
        "best_val_loss": float(best_val_loss),
        "last_val_loss": None if last_val_loss is None else float(last_val_loss),
    }


def save_last_checkpoint(checkpoint_dir, ckpt_dict):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "last.pt"
    torch.save(ckpt_dict, path)
    return path


def save_best_checkpoint(checkpoint_dir, ckpt_dict):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "best.pt"
    torch.save(ckpt_dict, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    validate_checkpoint_schema(ckpt)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    state = {
        "epoch": int(ckpt.get("epoch", 0)),
        "step": int(ckpt.get("step", 0)),
        "seed": int(ckpt.get("seed", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("inf"))),
        "last_val_loss": ckpt.get("last_val_loss", None),
        "raw": ckpt,
    }
    return state