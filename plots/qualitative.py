"""
Qualitative reconstruction figure — Figure 5.

Layout:
  Rows    = severity levels (default: 10%, 30%, 40%)
  Columns = Original | Masked Input | U-Net | Partial Conv | Gated Conv | Error Map

The error map shows absolute per-pixel error inside the masked region only,
rendered with a sequential colormap (brighter = larger error).

All three models are applied to the EXACT same image and mask.

This script imports from the main codebase (models, data) to run inference.
It does NOT modify any existing file.

Usage:
  python plots/qualitative.py \\
    --ckpts runs/unet__carpet.../checkpoints/best.pt \\
            runs/partial_conv__carpet.../checkpoints/best.pt \\
            runs/gated_conv__carpet.../checkpoints/best.pt \\
    --image_idx 0 \\
    --severities 10 30 40 \\
    --out figures/qualitative/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# Allow running from repo root
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from plots._utils import MODEL_LABELS, MODEL_ORDER, set_paper_style, savefig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _denorm(t: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Denormalize (C,H,W) or (1,C,H,W) tensor to [0,1]."""
    m = torch.tensor(mean, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    s = torch.tensor(std,  dtype=t.dtype, device=t.device).view(-1, 1, 1)
    if t.dim() == 4:
        m, s = m.unsqueeze(0), s.unsqueeze(0)
    return (t * s + m).clamp(0, 1)


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float tensor in [0,1] to HxWxC uint8."""
    return (t.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _make_block_mask(H: int, W: int, severity_pct: float, seed: int = 0) -> torch.Tensor:
    """
    Generate a centered square block mask with area ≈ severity_pct % of H*W.
    Returns (1, H, W) float32 tensor with 1 = masked.
    """
    area  = (severity_pct / 100.0) * H * W
    side  = max(1, min(int(round(area ** 0.5)), H, W))
    rng   = np.random.default_rng(seed)
    y0    = rng.integers(0, max(H - side, 0) + 1)
    x0    = rng.integers(0, max(W - side, 0) + 1)
    mask  = torch.zeros(1, H, W, dtype=torch.float32)
    mask[:, y0:y0 + side, x0:x0 + side] = 1.0
    return mask


def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Load model and normalization stats from a checkpoint."""
    from omegaconf import OmegaConf
    from models.build import build_model
    from training.checkpoint import validate_checkpoint_schema

    raw = torch.load(ckpt_path, map_location=device)
    validate_checkpoint_schema(raw)
    model_cfg   = OmegaConf.create(raw["model_cfg"])
    dataset_cfg = OmegaConf.create(raw["dataset_cfg"])

    model = build_model(model_cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()

    mean = list(dataset_cfg.norm.mean)
    std  = list(dataset_cfg.norm.std)
    return model, mean, std, dataset_cfg


def _load_image(dataset_cfg, image_idx: int, device: torch.device) -> torch.Tensor:
    """Load one image from the val split, return (1,C,H,W) normalized tensor."""
    from omegaconf import OmegaConf
    from data.build import build_base_dataset

    ds = build_base_dataset(dataset_cfg, split="val")
    item = ds[image_idx % len(ds)]
    if isinstance(item, dict):
        img = item["image"]
    else:
        img = item
    if not isinstance(img, torch.Tensor):
        from torchvision.transforms.functional import to_tensor
        img = to_tensor(img)
    return img.unsqueeze(0).to(device)


@torch.no_grad()
def _run_model(model, img_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return reconstruction (1,C,H,W) in normalized space."""
    masked = img_norm * (1.0 - mask)
    x      = torch.cat([masked, mask], dim=1)
    return model(x)


def _error_map(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    Compute per-pixel mean absolute error inside the masked region.
    Returns (H,W) float32 array in [0,1], with 0 outside mask.
    """
    err = (pred - gt).abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
    err = (err * mask).squeeze().cpu().numpy()          # (H,W)
    m   = mask.squeeze().cpu().numpy()
    if m.sum() > 0:
        mx = err.max()
        if mx > 1e-8:
            err = err / mx
    return err.astype(np.float32)


# ── Figure assembly ───────────────────────────────────────────────────────────

def build_figure(
    rows: list[dict],   # each: {severity, img_np, masked_np, recons: {model: np}, err_np}
    out_dir: Path,
) -> None:
    """
    Compose the qualitative figure.
    Columns: Original | Masked | U-Net | Partial Conv | Gated Conv | Error Map
    """
    n_rows = len(rows)
    n_cols = 6
    col_titles = [
        "Original", "Masked Input",
        MODEL_LABELS["unet"], MODEL_LABELS["partial_conv"], MODEL_LABELS["gated_conv"],
        "Error Map",
    ]

    fig_w = n_cols * 1.2
    fig_h = n_rows * 1.2 + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.04, wspace=0.03,
        top=0.94, bottom=0.01, left=0.07, right=0.99,
    )

    for r_idx, row in enumerate(rows):
        images = [
            row["img_np"],
            row["masked_np"],
            row["recons"].get("unet"),
            row["recons"].get("partial_conv"),
            row["recons"].get("gated_conv"),
            None,   # error map handled separately
        ]

        for c_idx in range(n_cols):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c_idx == 0:
                ax.set_ylabel(f"{int(row['severity'])}%", fontsize=7,
                              rotation=0, labelpad=22, va="center")

            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=7, pad=3)

            if c_idx < 5:
                im_data = images[c_idx]
                if im_data is not None:
                    ax.imshow(im_data)
                else:
                    ax.set_facecolor("#dddddd")
            else:
                # Error map
                err = row.get("err_np")
                if err is not None:
                    ax.imshow(err, cmap="hot", vmin=0, vmax=1)
                else:
                    ax.set_facecolor("#dddddd")

    savefig(fig, out_dir, "qualitative")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Qualitative reconstruction figure with error maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ckpts", nargs="+", required=True,
        help="Checkpoint paths, one per model. "
             "Order is inferred from the 'model' field inside each checkpoint.",
    )
    ap.add_argument("--image_idx", type=int, default=0,
                    help="Index of the image in the val set to use.")
    ap.add_argument("--severities", nargs="+", type=float, default=[10.0, 30.0, 40.0],
                    help="Mask severity percentages to visualize.")
    ap.add_argument("--mask_seed", type=int, default=7,
                    help="Seed for block mask placement (same across all models).")
    ap.add_argument("--out", default="figures/qualitative")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir = Path(args.out)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── Load all checkpoints ─────────────────────────────────────────────────
    models_by_name: dict[str, tuple] = {}
    mean_std = None
    dataset_cfg_ref = None

    for ckpt_path in args.ckpts:
        print(f"  loading {ckpt_path} ...")
        model, mean, std, dataset_cfg = _load_model_from_ckpt(ckpt_path, device)
        model_name = model.__class__.__name__.lower()
        # Normalise name to match MODEL_ORDER keys
        for key in MODEL_ORDER:
            if key.replace("_", "") in model_name.replace("_", ""):
                model_name = key
                break
        models_by_name[model_name] = (model, mean, std)
        if mean_std is None:
            mean_std = (mean, std)
            dataset_cfg_ref = dataset_cfg
        print(f"    model_name={model_name}")

    if not models_by_name:
        sys.exit("ERROR: no checkpoints loaded.")
    if mean_std is None:
        sys.exit("ERROR: could not read normalization stats.")

    mean, std = mean_std

    # ── Load image ───────────────────────────────────────────────────────────
    print(f"  loading image idx={args.image_idx} ...")
    img_norm = _load_image(dataset_cfg_ref, args.image_idx, device)  # (1,C,H,W)
    _, C, H, W = img_norm.shape
    img_vis = _to_np(_denorm(img_norm[0], mean, std))

    # ── Build rows ───────────────────────────────────────────────────────────
    rows = []
    for sev in sorted(args.severities):
        mask = _make_block_mask(H, W, sev, seed=args.mask_seed).to(device)
        mask_b = mask.unsqueeze(0)                    # (1,1,H,W)
        masked_vis = _to_np(_denorm((img_norm * (1.0 - mask_b))[0], mean, std))

        recons: dict[str, np.ndarray] = {}
        err_maps: list[np.ndarray]    = []

        for model_name in MODEL_ORDER:
            entry = models_by_name.get(model_name)
            if entry is None:
                print(f"  WARNING: no checkpoint for {model_name}, skipping")
                continue
            model, _, _ = entry
            pred = _run_model(model, img_norm, mask_b)           # (1,C,H,W) normalized
            recon_vis = _to_np(_denorm(pred[0], mean, std))
            recons[model_name] = recon_vis
            err_maps.append(_error_map(pred[0], img_norm[0], mask[0]))

        # Average error map across models
        avg_err = np.mean(err_maps, axis=0) if err_maps else None

        rows.append({
            "severity": sev,
            "img_np":    img_vis,
            "masked_np": masked_vis,
            "recons":    recons,
            "err_np":    avg_err,
        })
        print(f"  severity={sev}%  models={list(recons.keys())}")

    # ── Render ───────────────────────────────────────────────────────────────
    build_figure(rows, out_dir)
    print(f"\nDone. Figure saved to {out_dir}/")


if __name__ == "__main__":
    main()
