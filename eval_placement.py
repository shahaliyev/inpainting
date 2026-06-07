"""
Per-image placement evaluation for inpainting checkpoints.

Evaluates fixed square block masks at five spatial positions and configurable
severity levels. Writes one CSV row per (image × severity × placement) condition,
suitable for downstream statistical analysis.

Does NOT touch train.py, eval.py, degradation_v1, or paper plotting tools.

Usage
-----
Single checkpoint:
    python eval_placement.py \\
        --ckpt runs/<train_run>/checkpoints/best.pt

Scan all best.pt checkpoints under runs/ (default scan dir):
    python eval_placement.py --scan_runs

Scan a custom directory:
    python eval_placement.py --scan_runs my_runs/

Quick test (50 images, no LPIPS):
    python eval_placement.py \\
        --ckpt runs/<train_run>/checkpoints/best.pt \\
        --n_images 50 --no_lpips

Override severities / placements:
    python eval_placement.py \\
        --ckpt runs/.../best.pt \\
        --severities 10 30 --placements center top_left

Output
------
    runs/placement/<run_name>.csv   ← per-image scores
    runs/placement/<run_name>.json  ← run metadata sidecar

Override output directory:
    --out_dir path/to/dir

Override CSV path explicitly (single-checkpoint mode only):
    --out path/to/results.csv

CSV columns
-----------
    model, domain, image, mask_severity, mask_placement,
    l1, psnr, ssim, lpips
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from data.build import build_base_dataset
from mask.placed_block import PLACEMENTS, make_placed_block_mask
from models.build import build_model
from training.checkpoint import validate_checkpoint_schema
from utils.config_resolver import require_cfg_fields

DEFAULT_EVAL_CFG = "configs/eval/placement_v1.yaml"
DEFAULT_SEVERITIES = [5, 10, 20, 30, 40]
DEFAULT_PLACEMENTS = list(PLACEMENTS)
DEFAULT_N_IMAGES = 1000
DEFAULT_OUT_DIR = "runs/placement"

CSV_COLS = [
    "model", "domain", "image",
    "mask_severity", "mask_placement",
    "l1", "psnr", "ssim", "lpips",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-image placement evaluation of inpainting checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ckpt",
        help="Path to a single v1 checkpoint (.pt).",
    )
    src.add_argument(
        "--scan_runs",
        nargs="?",
        const="runs",
        metavar="DIR",
        help=(
            "Scan DIR for all best.pt checkpoints and evaluate each one. "
            "Defaults to 'runs/' when no DIR is given."
        ),
    )

    ap.add_argument("--eval_cfg", default=DEFAULT_EVAL_CFG,
                    help="Placement eval config YAML.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="Dataset split to evaluate.")
    ap.add_argument("--n_images", type=int, default=None,
                    help="Images per domain. Overrides eval_cfg.n_images.")
    ap.add_argument("--severities", type=int, nargs="+", default=None,
                    metavar="PCT",
                    help="Mask severities in %%. Overrides eval_cfg.severities.")
    ap.add_argument("--placements", nargs="+", default=None,
                    choices=list(PLACEMENTS), metavar="POS",
                    help="Placements to run. Overrides eval_cfg.placements.")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override loader batch size from checkpoint.")
    ap.add_argument("--metric_scope", choices=["mask", "full"], default="mask",
                    help="Region over which metrics are computed.")
    ap.add_argument("--no_lpips", action="store_true",
                    help="Skip LPIPS computation (recommended for CPU runs).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for dataset shuffling / reproducibility.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                    help="Root directory for CSV and JSON outputs.")
    ap.add_argument("--out", default=None,
                    help=(
                        "Explicit CSV output path (single --ckpt mode only). "
                        "A .json sidecar is written alongside it automatically."
                    ))
    return ap.parse_args()


# ── Checkpoint / path helpers ─────────────────────────────────────────────────

def _collect_ckpts(args) -> list[Path]:
    if args.scan_runs is not None:
        scan_dir = Path(args.scan_runs)
        if not scan_dir.exists():
            raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
        ckpts = sorted(scan_dir.glob("*/checkpoints/best.pt"))
        if not ckpts:
            raise FileNotFoundError(
                f"No best.pt checkpoints found under: {scan_dir}"
            )
        return ckpts
    p = Path(args.ckpt)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return [p]


def _out_paths(
    ckpt_path: Path,
    out_dir: Path,
    explicit_out: str | None,
) -> tuple[Path, Path]:
    """Return (csv_path, json_path) for this checkpoint's outputs."""
    if explicit_out:
        csv_path = Path(explicit_out)
    else:
        run_name = ckpt_path.parent.parent.name
        csv_path = out_dir / f"{run_name}.csv"
    json_path = csv_path.with_suffix(".json")
    return csv_path, json_path


def _cfg_from_ckpt(ckpt_raw: dict, key: str):
    if key not in ckpt_raw or ckpt_raw[key] is None:
        raise ValueError(
            f"Checkpoint is missing required key '{key}'. "
            "Re-train with the latest train.py."
        )
    return OmegaConf.create(ckpt_raw[key])


# ── Denormalisation ───────────────────────────────────────────────────────────

def _denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x * std + mean).clamp(0.0, 1.0)


# ── Per-image metrics ─────────────────────────────────────────────────────────

@torch.no_grad()
def _per_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lpips_net=None,
    scope: str = "mask",
) -> list[dict]:
    """
    Compute per-image metrics for a batch.

    Args:
        pred:      Model output, normalised space  (B, 3, H, W).
        target:    Ground-truth image, normalised  (B, 3, H, W).
        mask:      Binary hole mask, 1 = masked    (B, 1, H, W).
        mean/std:  Normalisation statistics         (1, 3, 1, 1).
        lpips_net: Optional LPIPS network; skipped when None.
        scope:     'mask' — metrics over the hole only.
                   'full' — metrics over the full reconstructed image.

    Returns:
        List of dicts, one per image: keys l1, psnr, ssim, lpips.
        lpips is None when lpips_net is None.
    """
    # Cast to float32; AMP bfloat16 must not affect metric values.
    pred   = pred.float()
    target = target.float()
    mask   = mask.float()

    C = pred.shape[1]

    pred_01  = _denorm(pred,  mean, std)
    target_01 = _denorm(target, mean, std)
    recon    = target * (1.0 - mask) + pred * mask
    recon_01 = _denorm(recon, mean, std)

    rows = []
    for i in range(pred.shape[0]):
        p_n  = pred[i:i+1];   t_n  = target[i:i+1]   # normalised (1, C, H, W)
        p_01 = pred_01[i];    t_01 = target_01[i]     # [0,1]      (C, H, W)
        r_01 = recon_01[i]
        m_b  = mask[i:i+1]                            # (1, 1, H, W)
        m    = mask[i]                                 # (1, H, W)

        # L1 (in normalised space, matching engine.py convention)
        if scope == "mask":
            l1 = ((torch.abs(p_n - t_n) * m_b).sum() /
                  (m_b.sum() * C + 1e-8)).item()
        else:
            recon_n = t_n * (1.0 - m_b) + p_n * m_b
            l1 = torch.abs(recon_n - t_n).mean().item()

        # PSNR (in [0,1] space)
        if scope == "mask":
            se      = (p_01 - t_01).pow(2) * m
            mse_val = se.sum() / (m.sum() * C + 1e-8)
        else:
            mse_val = ((r_01 - t_01) ** 2).mean()
        psnr = float(-10.0 * torch.log10(mse_val + 1e-10).item())

        # SSIM
        r_np = r_01.permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
        t_np = t_01.permute(1, 2, 0).cpu().numpy()
        if scope == "mask":
            m_np = m[0].cpu().numpy().astype(np.float32)
            _, ssim_map = ssim_fn(
                r_np, t_np, data_range=1.0, channel_axis=2, full=True
            )
            if ssim_map.ndim == 3:
                ssim_map = ssim_map.mean(axis=2)
            ssim_val = float((ssim_map * m_np).sum() / (m_np.sum() + 1e-8))
        else:
            ssim_val = float(ssim_fn(r_np, t_np, data_range=1.0, channel_axis=2))

        # LPIPS
        lpips_val = None
        if lpips_net is not None:
            if scope == "mask":
                d = lpips_net(
                    p_01.unsqueeze(0) * m_b,
                    t_01.unsqueeze(0) * m_b,
                    normalize=True,
                )
            else:
                d = lpips_net(r_01.unsqueeze(0), t_01.unsqueeze(0), normalize=True)
            lpips_val = float(d.item())

        rows.append({"l1": l1, "psnr": psnr, "ssim": ssim_val, "lpips": lpips_val})

    return rows


# ── Single-checkpoint evaluation ──────────────────────────────────────────────

def run_one(
    ckpt_path: Path,
    args,
    severities: list,
    placements: list,
    n_images: int,
    out_dir: Path,
    device: torch.device,
    lpips_net,
) -> None:
    """Load one checkpoint and run the full placement evaluation."""

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    validate_checkpoint_schema(ckpt_raw)

    model_cfg   = _cfg_from_ckpt(ckpt_raw, "model_cfg")
    dataset_cfg = _cfg_from_ckpt(ckpt_raw, "dataset_cfg")
    loader_cfg  = _cfg_from_ckpt(ckpt_raw, "loader_cfg")
    train_cfg   = _cfg_from_ckpt(ckpt_raw, "train_cfg")

    require_cfg_fields(dataset_cfg, ["norm.mean", "norm.std"], "checkpoint dataset_cfg")
    require_cfg_fields(loader_cfg,  ["batch_size"],            "checkpoint loader_cfg")

    state_epoch = int(ckpt_raw.get("epoch", 0))
    state_step  = int(ckpt_raw.get("step",  0))

    if "${oc.env:" in str(getattr(dataset_cfg, "root", "")):
        OmegaConf.resolve(dataset_cfg)

    config_paths = ckpt_raw.get("config_paths", {}) or {}
    model_name   = Path(config_paths.get("model_yaml",   "unknown")).stem
    domain_name  = Path(config_paths.get("dataset_yaml", "unknown")).stem

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt_raw["model"])
    model.eval()

    use_amp   = bool(getattr(train_cfg, "mixed_precision", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    mean = torch.tensor(list(dataset_cfg.norm.mean), device=device).view(1, 3, 1, 1)
    std  = torch.tensor(list(dataset_cfg.norm.std),  device=device).view(1, 3, 1, 1)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
    ds_cfg.limit         = n_images
    ds_cfg.limit_shuffle = False
    ds_cfg.limit_seed    = args.seed

    base_ds  = build_base_dataset(ds_cfg, split=args.split)
    actual_n = len(base_ds)
    if actual_n < n_images:
        print(f"  WARNING: only {actual_n} images available (requested {n_images}).")

    batch_size  = args.batch_size or int(loader_cfg.batch_size)
    num_workers = int(getattr(loader_cfg, "num_workers", 0))
    dl = DataLoader(
        base_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    explicit_out = args.out if args.scan_runs is None else None
    csv_path, json_path = _out_paths(ckpt_path, out_dir, explicit_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_conditions = len(severities) * len(placements)
    total_rows   = actual_n * n_conditions

    print(
        f"\n  checkpoint  : {ckpt_path}"
        f"\n  model       : {model_name}   domain: {domain_name}"
        f"\n  epoch={state_epoch}  step={state_step}  split={args.split}"
        f"\n  images      : {actual_n}   batch_size: {batch_size}"
        f"\n  conditions  : {n_conditions}   total_rows: {total_rows}"
        f"\n  output      : {csv_path}\n"
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
        writer.writeheader()

        cond_idx = 0
        for severity in severities:
            for placement in placements:
                cond_idx += 1
                mask_template: torch.Tensor | None = None
                img_count = 0

                for batch in dl:
                    imgs  = batch["image"].to(device, non_blocking=True)
                    paths = batch["path"]
                    B, _, H, W = imgs.shape

                    if mask_template is None:
                        # Pure function of (H, W, severity, placement) — no randomness.
                        # Identical across models for the same condition.
                        mask_template = make_placed_block_mask(
                            H, W, float(severity), placement
                        ).to(device)

                    mask_b = mask_template.unsqueeze(0).expand(B, -1, -1, -1)
                    x = torch.cat([imgs * (1.0 - mask_b), mask_b], dim=1)

                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        pred = model(x)

                    metrics_list = _per_image_metrics(
                        pred, imgs, mask_b, mean, std,
                        lpips_net=lpips_net,
                        scope=args.metric_scope,
                    )

                    for img_path, m in zip(paths, metrics_list):
                        writer.writerow({
                            "model":          model_name,
                            "domain":         domain_name,
                            "image":          img_path,
                            "mask_severity":  severity,
                            "mask_placement": placement,
                            "l1":             f"{m['l1']:.8f}",
                            "psnr":           f"{m['psnr']:.6f}",
                            "ssim":           f"{m['ssim']:.6f}",
                            "lpips": (
                                f"{m['lpips']:.8f}"
                                if m["lpips"] is not None else ""
                            ),
                        })

                    img_count += B

                fh.flush()
                print(
                    f"    [{cond_idx:2d}/{n_conditions}]  "
                    f"severity={severity:2d}%  "
                    f"placement={placement:<13s}  "
                    f"images={img_count}"
                )

    # ── Metadata sidecar ──────────────────────────────────────────────────────
    meta = {
        "ckpt":         str(ckpt_path.resolve()),
        "run_name":     ckpt_path.parent.parent.name,
        "model":        model_name,
        "domain":       domain_name,
        "epoch":        state_epoch,
        "step":         state_step,
        "split":        args.split,
        "n_images":     actual_n,
        "severities":   severities,
        "placements":   placements,
        "metric_scope": args.metric_scope,
        "lpips":        lpips_net is not None,
        "eval_cfg":     args.eval_cfg,
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    print(f"\n  Done. CSV  → {csv_path}  ({total_rows} rows)")
    print(f"        JSON → {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.out and args.scan_runs is not None:
        raise ValueError("--out cannot be combined with --scan_runs. Use --out_dir instead.")

    torch.manual_seed(args.seed)

    # ── Eval protocol (loaded once, shared across all checkpoints) ────────────
    eval_cfg_path = Path(args.eval_cfg)
    if not eval_cfg_path.exists():
        raise FileNotFoundError(f"Eval config not found: {eval_cfg_path}")
    eval_cfg = OmegaConf.load(eval_cfg_path)

    severities = args.severities or list(eval_cfg.get("severities", DEFAULT_SEVERITIES))
    placements = args.placements or list(eval_cfg.get("placements", DEFAULT_PLACEMENTS))
    n_images   = args.n_images   or int(eval_cfg.get("n_images",   DEFAULT_N_IMAGES))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpts = _collect_ckpts(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── LPIPS (loaded once, reused across all checkpoints) ───────────────────
    if args.no_lpips:
        lpips_net = None
    else:
        if device.type == "cpu":
            print("WARNING: LPIPS on CPU is slow. Pass --no_lpips to skip it.")
        import lpips as lpips_lib
        lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()

    print(
        f"eval_cfg    : {eval_cfg_path}"
        f"\nseverities  : {severities}"
        f"\nplacements  : {placements}"
        f"\nn_images    : {n_images}"
        f"\nmetric_scope: {args.metric_scope}   lpips: {lpips_net is not None}"
        f"\nout_dir     : {out_dir}"
        f"\ncheckpoints : {len(ckpts)}"
    )

    for i, ckpt_path in enumerate(ckpts):
        if len(ckpts) > 1:
            print(f"\n[{i + 1}/{len(ckpts)}] {ckpt_path.parent.parent.name}")
        run_one(
            ckpt_path=ckpt_path,
            args=args,
            severities=severities,
            placements=placements,
            n_images=n_images,
            out_dir=out_dir,
            device=device,
            lpips_net=lpips_net,
        )

    if len(ckpts) > 1:
        print(f"\nAll done. {len(ckpts)} checkpoints evaluated → {out_dir}/")


if __name__ == "__main__":
    main()
