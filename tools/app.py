import base64
import io
import random
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build import build_base_dataset
from mask.build import build_mask_generator
from models.build import build_model
from training.checkpoint import validate_checkpoint_schema
from utils.config_resolver import resolve_config_path
from utils.metrics import compute_metrics


DATASET_OPTIONS = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
MODEL_OPTIONS = ["gated_conv", "unet", "partial_conv"]
MASK_OPTIONS = ["block", "multi_block", "freeform"]


# ── CSS ──────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* global */
        .block-container {padding-top: 3rem !important; padding-bottom: 1.5rem !important;}
        .app-subtitle {color: #64748b; font-size: 0.82rem; margin: 1px 0 14px 0;}

        /* sidebar compactness */
        section[data-testid="stSidebar"] > div {padding-top: 0.7rem !important;}
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stCheckbox {margin-bottom: 0 !important;}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stNumberInput label,
        section[data-testid="stSidebar"] .stCheckbox label {
            font-size: 0.76rem !important; color: #475569 !important;
            margin-bottom: 1px !important;
        }

        /* sidebar meta text */
        .sb-section {font-size: 0.7rem; font-weight: 700; color: #94a3b8;
            text-transform: uppercase; letter-spacing: 0.06em;
            margin: 8px 0 3px 0;}
        .sb-divider {border:none; border-top:1px solid #e2e8f0; margin: 8px 0;}
        .stats-line {font-size: 0.74rem; color: #64748b; margin: 2px 0 6px 0;}
        .ckpt-found  {font-size: 0.74rem; color: #16a34a; font-weight: 700; margin: 2px 0 2px 0;}
        .ckpt-missing{font-size: 0.74rem; color: #dc2626; font-weight: 700; margin: 2px 0 2px 0;}
        .ckpt-path   {font-size: 0.68rem; color: #94a3b8; word-break:break-all;
            margin: 0 0 8px 0; line-height: 1.35;}

        /* chips */
        .chip-row {display:flex; flex-wrap:wrap; gap:5px; margin: 6px 0 14px 0;}
        .chip {padding:2px 8px; border-radius:999px; font-size:0.72rem;
            border:1px solid #bfdbfe; color:#1e40af; background:#eff6ff;
            white-space:nowrap;}

        /* section headers */
        .sec-header {font-size:0.85rem; font-weight:700; color:#1e293b;
            text-transform:uppercase; letter-spacing:0.04em;
            border-bottom:2px solid #e2e8f0; padding-bottom:4px;
            margin: 20px 0 10px 0;}

        /* pair */
        .pair-meta {font-size:0.75rem; color:#475569; margin:0 0 5px 0; line-height:1.6;}
        .img-label {font-size:0.67rem; font-weight:700; text-transform:uppercase;
            letter-spacing:0.07em; color:#94a3b8; margin-bottom:2px;}

        /* summary stats */
        .summary-box {background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
            padding:10px 14px; margin: 10px 0;}
        .summary-row {display:flex; gap:24px; flex-wrap:wrap;}
        .stat-item {text-align:center;}
        .stat-val {font-size:1.05rem; font-weight:700; color:#1e293b;}
        .stat-lbl {font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;}

        /* callout badges */
        .badge-best  {background:#dcfce7; color:#15803d; border:1px solid #86efac;
            border-radius:6px; padding:1px 7px; font-size:0.7rem; font-weight:700;}
        .badge-worst {background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5;
            border-radius:6px; padding:1px 7px; font-size:0.7rem; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset_cfg(dataset_key: str):
    dataset_yaml = resolve_config_path("dataset", dataset_key)
    dataset_cfg = OmegaConf.load(dataset_yaml)
    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError(f"dataset_cfg.root is missing in {dataset_yaml}")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)
    return dataset_cfg


def _load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    validate_checkpoint_schema(ckpt)
    model_cfg = OmegaConf.create(ckpt["model_cfg"])
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def _build_mask_cfg(mask_name: str, ratios: list, freeform_strokes: int, seed: int):
    cfg = {"name": mask_name, "train": {"deterministic": False}, "eval": {"deterministic": False}}
    if mask_name in {"block", "multi_block"}:
        if not ratios:
            raise ValueError("At least one ratio must be provided.")
        cfg["ratios"] = [float(r) for r in ratios]
    if mask_name == "multi_block":
        cfg.update({"min_blocks": 2, "max_blocks": 5})
    if mask_name == "freeform":
        cfg.update({
            "num_strokes": int(freeform_strokes), "min_vertices": 4, "max_vertices": 12,
            "min_brush_width": 8, "max_brush_width": 30,
            "max_angle": 0.7853981633974483, "min_length": 20, "max_length": 80,
        })
    return OmegaConf.create(cfg), seed


def _denorm_to_nhwc(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()


def _apply_black_mask(original: np.ndarray, mask_1hw: torch.Tensor) -> np.ndarray:
    out = original.copy()
    out[mask_1hw[0].detach().cpu().numpy() > 0.5] = 0.0
    return out


def _diff_heatmap(original: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Absolute difference averaged over channels, returned as HxWx3 RGB heatmap."""
    diff = np.abs(original.astype(np.float32) - recon.astype(np.float32)).mean(axis=2)
    normed = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    rgba = cm.hot(normed)
    return rgba[:, :, :3].astype(np.float32)


def _mask_crop_zoom(
    orig: np.ndarray,
    recon: np.ndarray,
    mask_1hw: torch.Tensor,
    pad_factor: float = 0.5,
):
    """Crop both images to the bounding box of the mask region with padding."""
    mask_np = mask_1hw[0].detach().cpu().numpy() > 0.5
    rows_hit = np.any(mask_np, axis=1)
    cols_hit = np.any(mask_np, axis=0)
    if not rows_hit.any():
        return orig, recon
    r0 = int(np.where(rows_hit)[0][0])
    r1 = int(np.where(rows_hit)[0][-1])
    c0 = int(np.where(cols_hit)[0][0])
    c1 = int(np.where(cols_hit)[0][-1])
    h, w = mask_np.shape
    pad_r = max(8, int((r1 - r0 + 1) * pad_factor))
    pad_c = max(8, int((c1 - c0 + 1) * pad_factor))
    r0c = max(0, r0 - pad_r)
    r1c = min(h, r1 + pad_r + 1)
    c0c = max(0, c0 - pad_c)
    c1c = min(w, c1 + pad_c + 1)
    return orig[r0c:r1c, c0c:c1c], recon[r0c:r1c, c0c:c1c]


def _parse_ratios(raw: str) -> list:
    vals = sorted({float(t.strip()) for t in raw.split(",") if t.strip()})
    if not vals:
        raise ValueError("No valid ratios.")
    return vals


def _parse_pixels(raw: str) -> list:
    vals = sorted({float(t.strip()) for t in raw.split(",") if t.strip()})
    if not vals:
        raise ValueError("No valid pixel values.")
    return vals


def _ratios_to_pixels(ratios: list, h: int, w: int) -> list:
    base = float(max(1, min(h, w)))
    return [max(1.0, (r / 100.0) * base) for r in ratios]


def _pixels_to_ratios(pixels: list, h: int, w: int) -> list:
    base = float(max(1, min(h, w)))
    return sorted({max(0.1, (px / base) * 100.0) for px in pixels})


def _ui_ratio_to_area_ratio(ui_ratio: float, h: int, w: int) -> float:
    """
    Convert side-based UI ratio (% of min(H,W)) to area-based ratio (% of H*W).
    Kept as float so the mask generator receives meaningful values.
    e.g. 20% side on 256x256 -> 51px side -> 2621px² / 65536 = 4.0% area.
    """
    side_px = max(1.0, (ui_ratio / 100.0) * float(min(h, w)))
    area_pct = (side_px * side_px) / float(max(1, h * w)) * 100.0
    return max(0.1, area_pct)


def _find_best_ckpt(dataset_key: str, model_key: str) -> "Path | None":
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    prefix = f"{model_key}__{dataset_key}__"
    candidates = [
        r / "checkpoints" / "best.pt"
        for r in runs_dir.iterdir()
        if r.is_dir() and r.name.startswith(prefix) and (r / "checkpoints" / "best.pt").exists()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


@st.cache_data(show_spinner=False)
def _dataset_stats(dataset_key: str, split: str) -> dict:
    dataset_cfg = _load_dataset_cfg(dataset_key)
    ds_split = build_base_dataset(dataset_cfg, split=split)
    n = len(ds_split)
    if n == 0:
        return {"count": 0, "h": None, "w": None, "train_count": 0, "val_count": 0}
    n_train = len(build_base_dataset(dataset_cfg, split="train"))
    n_val = len(build_base_dataset(dataset_cfg, split="val"))
    img = ds_split[0]["image"]
    if not torch.is_tensor(img) or img.ndim != 3:
        return {"count": n, "h": None, "w": None, "train_count": n_train, "val_count": n_val}
    _, h, w = img.shape
    return {"count": n, "h": int(h), "w": int(w), "train_count": n_train, "val_count": n_val}


def _build_mean_std(dataset_cfg, device):
    mean = torch.tensor(list(dataset_cfg.norm.mean), dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(list(dataset_cfg.norm.std), dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def _sample_single(ds, seed: int):
    if len(ds) == 0:
        raise ValueError("Dataset split is empty.")
    rng = random.Random(seed)
    idx = rng.randrange(0, len(ds))
    return ds[idx], idx


# ── inference pipeline ────────────────────────────────────────────────────────

def _run_inference(
    ckpt_path: str, dataset_key: str, split: str,
    mask_name: str, ratios: list, freeform_strokes: int,
    n_masks: int, seed: int, metric_scope: str, use_lpips: bool,
) -> dict:
    status = st.empty()

    status.info("⏳ Loading model…")
    model, device = _load_model_from_ckpt(ckpt_path)

    status.info("⏳ Loading dataset…")
    dataset_cfg = _load_dataset_cfg(dataset_key)
    ds = build_base_dataset(dataset_cfg, split=split)

    status.info("⏳ Sampling image…")
    sample, sampled_idx = _sample_single(ds, seed)
    image = sample["image"]
    if not torch.is_tensor(image) or image.ndim != 3:
        raise ValueError("Expected CHW tensor from dataset transform.")
    path = sample.get("path", "unknown")

    status.info("⏳ Generating masks…")
    img_b = image.unsqueeze(0).to(device)
    masks, masked_imgs, ui_ratios_used = [], [], []
    if ratios:
        # block / multi_block: one dedicated generator per ratio → deterministic assignment
        for i, r in enumerate(ratios):
            single_cfg, mask_seed_i = _build_mask_cfg(mask_name, [r], freeform_strokes, seed + i)
            gen = build_mask_generator(
                single_cfg, split="eval",
                train_seed=mask_seed_i, eval_seed=mask_seed_i + 1,
            )
            m = gen(image.shape)
            masks.append(m)
            masked_imgs.append(image * (1.0 - m))
            ui_ratios_used.append(float(r))
    else:
        # freeform: all masks from one shared generator
        mask_cfg, mask_seed = _build_mask_cfg(mask_name, ratios, freeform_strokes, seed)
        mask_gen = build_mask_generator(
            mask_cfg, split="eval",
            train_seed=mask_seed, eval_seed=mask_seed + 1,
        )
        for _ in range(n_masks):
            m = mask_gen(image.shape)
            masks.append(m)
            masked_imgs.append(image * (1.0 - m))
            ui_ratios_used.append(None)

    n_variants = len(masks)
    mask_b = torch.stack(masks).to(device)
    masked_b = torch.stack(masked_imgs).to(device)
    img_rep = img_b.repeat(n_variants, 1, 1, 1)

    status.info("⏳ Running inference…")
    with torch.no_grad():
        pred = model(torch.cat([masked_b, mask_b], dim=1))
        recon = masked_b * (1.0 - mask_b) + pred * mask_b

    mean, std = _build_mean_std(dataset_cfg, device)
    mean3, std3 = mean[0], std[0]
    vis_orig = _denorm_to_nhwc(img_rep[0], mean3, std3)
    vis_masked = [_apply_black_mask(vis_orig, mask_b[i]) for i in range(n_variants)]
    vis_recon = [_denorm_to_nhwc(recon[i], mean3, std3) for i in range(n_variants)]
    vis_diff = [_diff_heatmap(vis_orig, vis_recon[i]) for i in range(n_variants)]
    vis_crops = [_mask_crop_zoom(vis_orig, vis_recon[i], mask_b[i]) for i in range(n_variants)]

    lpips_net = None
    if use_lpips:
        status.info("⏳ Loading LPIPS…")
        try:
            import lpips
            lpips_net = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            st.warning(f"LPIPS failed: {e}. Continuing without it.")
            use_lpips = False

    status.info("⏳ Computing metrics…")
    rows = []
    for i in range(n_variants):
        m = compute_metrics(
            pred=pred[i:i+1], target=img_rep[i:i+1], mask=mask_b[i:i+1],
            mean=mean, std=std, lpips_net=lpips_net,
            metric_scope=metric_scope, report_both=False,
        )
        row = {
            "variant": i + 1,
            "ui_ratio": ui_ratios_used[i],
            "mask_area_pct": float(mask_b[i].mean().item() * 100.0),
            "l1": float(m.get("l1", 0.0)),
            "psnr": float(m.get("psnr", 0.0)),
            "ssim": float(m.get("ssim", 0.0)),
        }
        if use_lpips:
            row["lpips"] = float(m.get("lpips", 0.0))
        rows.append(row)

    status.success("✓ Done")
    return {
        "sample_index": sampled_idx,
        "sample_path": path,
        "image_shape": tuple(image.shape),
        "original": vis_orig,
        "masked": vis_masked,
        "recon": vis_recon,
        "diff": vis_diff,
        "crops": vis_crops,
        "metrics": rows,
    }


# ── analysis charts ───────────────────────────────────────────────────────────

_HIGHER_BETTER = {"psnr", "ssim"}
_METRIC_ARROW  = {m: ("↑ higher better", "#16a34a") for m in _HIGHER_BETTER}
_METRIC_ARROW.update({m: ("↓ lower better", "#dc2626") for m in {"l1", "lpips"}})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _prepare_chart_df(rows: list):
    """Sort rows and choose X-axis values for metric-vs-area plots."""
    df = pd.DataFrame(rows)
    has_ui = "ui_ratio" in df.columns and df["ui_ratio"].notna().all()
    if has_ui:
        df = df.sort_values("ui_ratio").reset_index(drop=True)
        x_vals = df["ui_ratio"].tolist()
        x_label = "Input ratio (side % of min side)"
    else:
        df = df.sort_values("mask_area_pct").reset_index(drop=True)
        x_vals = df["mask_area_pct"].tolist()
        x_label = "Mask area %"
    return df, x_vals, x_label


def _draw_metric_line_axes(axes, df, x_vals, x_label, metrics, colors):
    """Shared drawing logic for line charts (used by both st display and HTML export)."""
    for j, (ax, metric) in enumerate(zip(axes, metrics)):
        ax.set_facecolor("#f8fafc")
        y_vals = df[metric].tolist()
        ax.plot(x_vals, y_vals, "o-", color=colors.get(metric, "#64748b"),
                linewidth=1.8, markersize=5, zorder=3)
        # degradation slope as dashed fit line
        if len(x_vals) >= 2:
            x_arr = np.array(x_vals, dtype=float)
            slope, intercept = np.polyfit(x_arr, y_vals, 1)
            fit_x = np.linspace(x_arr.min(), x_arr.max(), 80)
            ax.plot(fit_x, slope * fit_x + intercept, "--", color="#94a3b8",
                    linewidth=1.0, zorder=2, label=f"slope {slope:+.3f}/unit")
            ax.legend(fontsize=6, loc="best", framealpha=0.7, edgecolor="#e2e8f0")
        arrow_text, arrow_color = _METRIC_ARROW.get(metric, ("", "#64748b"))
        ax.set_title(f"{metric.upper()}  {arrow_text}", fontsize=9,
                     color=arrow_color, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=8, color="#64748b")
        ax.set_ylabel(metric.upper(), fontsize=8, color="#64748b")
        ax.tick_params(labelsize=7, colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
        ax.grid(True, alpha=0.4, color="#e2e8f0", zorder=0)
        # stagger labels up/down to avoid overlap
        for k, (xv, yv) in enumerate(zip(x_vals, y_vals)):
            ax.annotate(
                f"V{int(df.loc[k, 'variant'])}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, 7 if k % 2 == 0 else -11),
                fontsize=6, ha="center", color="#475569",
            )


def _chart_metric_vs_area(rows: list, use_lpips: bool) -> None:
    df, x_vals, x_label = _prepare_chart_df(rows)
    metrics = ["psnr", "ssim", "l1"]
    if use_lpips and "lpips" in df.columns:
        metrics.append("lpips")
    colors = {"psnr": "#2563eb", "ssim": "#16a34a", "l1": "#dc2626", "lpips": "#9333ea"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 3.5), dpi=110)
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#ffffff")
    _draw_metric_line_axes(axes, df, x_vals, x_label, metrics, colors)
    fig.suptitle("Metric vs Mask Ratio", fontsize=10, color="#1e293b", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _chart_metric_bars(rows: list, use_lpips: bool) -> None:
    df = pd.DataFrame(rows)
    metrics = ["psnr", "ssim", "l1"]
    if use_lpips and "lpips" in df.columns:
        metrics.append("lpips")
    colors = {"psnr": "#2563eb", "ssim": "#16a34a", "l1": "#dc2626", "lpips": "#9333ea"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 3.3), dpi=110)
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#ffffff")
    x_labels = [
        f"V{r['variant']}\n{r['ui_ratio']:.0f}%" if r.get("ui_ratio") is not None
        else f"V{r['variant']}\n{r['mask_area_pct']:.1f}%"
        for r in rows
    ]
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#f8fafc")
        vals = df[metric].tolist()
        best_i = int(np.argmax(vals)) if metric in _HIGHER_BETTER else int(np.argmin(vals))
        bar_colors = [colors.get(metric, "#64748b")] * len(vals)
        bar_colors[best_i] = "#fbbf24"
        ax.bar(x_labels, vals, color=bar_colors, edgecolor="#e2e8f0", linewidth=0.5)
        arrow_text, arrow_color = _METRIC_ARROW.get(metric, ("", "#64748b"))
        ax.set_title(f"{metric.upper()}  {arrow_text}", fontsize=9,
                     color=arrow_color, fontweight="bold")
        ax.set_xlabel("Variant (ratio)", fontsize=8, color="#64748b")
        ax.set_ylabel(metric.upper(), fontsize=8, color="#64748b")
        ax.tick_params(labelsize=7, colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
        ax.grid(True, axis="y", alpha=0.4, color="#e2e8f0")
    fig.suptitle("Per-Variant Comparison  (gold = best)", fontsize=10, color="#1e293b", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _img_to_b64_png(arr) -> str:
    """Convert a uint8 HWC numpy array to a base64-encoded PNG data URI."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_line_chart_b64(rows: list, use_lpips: bool) -> str:
    df, x_vals, x_label = _prepare_chart_df(rows)
    metrics = ["psnr", "ssim", "l1"] + (["lpips"] if use_lpips and "lpips" in df.columns else [])
    colors = {"psnr": "#2563eb", "ssim": "#16a34a", "l1": "#dc2626", "lpips": "#9333ea"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 3.5), dpi=110)
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#ffffff")
    _draw_metric_line_axes(axes, df, x_vals, x_label, metrics, colors)
    fig.suptitle("Metric vs Mask Ratio", fontsize=10, color="#1e293b", y=1.02)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return "data:image/png;base64," + b64


def _build_bar_chart_b64(rows: list, use_lpips: bool) -> str:
    df = pd.DataFrame(rows)
    metrics = ["psnr", "ssim", "l1"] + (["lpips"] if use_lpips and "lpips" in df.columns else [])
    colors = {"psnr": "#2563eb", "ssim": "#16a34a", "l1": "#dc2626", "lpips": "#9333ea"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.8 * len(metrics), 3.3), dpi=110)
    if len(metrics) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#ffffff")
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#f8fafc")
        vals = df[metric].tolist()
        best_i = int(np.argmax(vals)) if metric in _HIGHER_BETTER else int(np.argmin(vals))
        bar_colors = [colors.get(metric, "#64748b")] * len(vals)
        bar_colors[best_i] = "#fbbf24"
        x_labels = [
            f"V{r['variant']}\n{r['ui_ratio']:.0f}%" if r.get("ui_ratio") is not None
            else f"V{r['variant']}\n{r['mask_area_pct']:.1f}%"
            for r in rows
        ]
        ax.bar(x_labels, vals, color=bar_colors, edgecolor="#e2e8f0", linewidth=0.5)
        arrow_text, arrow_color = _METRIC_ARROW.get(metric, ("", "#64748b"))
        ax.set_title(f"{metric.upper()}  {arrow_text}", fontsize=9, color=arrow_color, fontweight="bold")
        ax.set_xlabel("Variant (area %)", fontsize=8, color="#64748b")
        ax.tick_params(labelsize=7, colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
        ax.grid(True, axis="y", alpha=0.4, color="#e2e8f0")
    fig.suptitle("Per-Variant Comparison (gold = best)", fontsize=10, color="#1e293b", y=1.02)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return "data:image/png;base64," + b64


def _generate_html_report(out: dict, use_lpips: bool, meta: dict) -> str:
    """Produce a fully self-contained HTML report (all images base64-embedded)."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── images ────────────────────────────────────────────────────────────────
    orig_uri = _img_to_b64_png(out["original"])
    masked_uris = [_img_to_b64_png(m) for m in out["masked"]]
    recon_uris  = [_img_to_b64_png(r) for r in out["recon"]]
    diff_uris   = [_img_to_b64_png(d) for d in out["diff"]]
    crop_uris = [
        (_img_to_b64_png(oc), _img_to_b64_png(rc))
        for oc, rc in out.get("crops", [])
    ]

    # ── charts ────────────────────────────────────────────────────────────────
    line_uri = bar_uri = ""
    if len(out["metrics"]) > 1:
        line_uri = _build_line_chart_b64(out["metrics"], use_lpips)
        bar_uri  = _build_bar_chart_b64(out["metrics"], use_lpips)

    # ── metrics table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(out["metrics"])
    table_html = df.to_html(index=False, border=0, classes="metrics-table",
                             float_format=lambda x: f"{x:.4f}")

    # ── variant rows ──────────────────────────────────────────────────────────
    psnr_vals = [r["psnr"] for r in out["metrics"]]
    variant_rows_html = ""
    for i, row in enumerate(out["metrics"]):
        badge = ""
        if len(psnr_vals) > 1:
            if row["psnr"] == max(psnr_vals):
                badge = "<span class='badge-best'>best</span>"
            elif row["psnr"] == min(psnr_vals):
                badge = "<span class='badge-worst'>worst</span>"
        lpips_td = f"<td>{row['lpips']:.4f}</td>" if "lpips" in row else ""
        variant_rows_html += f"""
        <div class="variant-block">
          <div class="variant-header">
            Variant {row['variant']} {badge}
            &nbsp;·&nbsp; area {row['mask_area_pct']:.1f}%
            &nbsp;·&nbsp; L1 {row['l1']:.4f}
            &nbsp;·&nbsp; PSNR {row['psnr']:.2f} dB
            &nbsp;·&nbsp; SSIM {row['ssim']:.4f}
            {"&nbsp;·&nbsp; LPIPS " + f"{row['lpips']:.4f}" if 'lpips' in row else ""}
          </div>
          <div class="triplet">
            <div class="triplet-item"><div class="img-lbl">Masked</div><img src="{masked_uris[i]}"></div>
            <div class="triplet-item"><div class="img-lbl">Inpainted</div><img src="{recon_uris[i]}"></div>
            <div class="triplet-item"><div class="img-lbl">Difference</div><img src="{diff_uris[i]}"></div>
          </div>
          {"" if i >= len(crop_uris) else f'''<div class="crop-row">
            <div class="triplet-item" style="max-width:200px"><div class="img-lbl">Crop · Original</div><img src="{crop_uris[i][0]}"></div>
            <div class="triplet-item" style="max-width:200px"><div class="img-lbl">Crop · Inpainted</div><img src="{crop_uris[i][1]}"></div>
          </div>'''}
        </div>"""

    # ── chart section ─────────────────────────────────────────────────────────
    chart_section = ""
    if line_uri:
        chart_section = f"""
        <h2>Analysis Plots</h2>
        <h3>Metric vs Mask Area</h3>
        <img src="{line_uri}" style="max-width:100%;">
        <h3>Per-Variant Comparison</h3>
        <img src="{bar_uri}" style="max-width:100%;">"""

    # ── assemble ──────────────────────────────────────────────────────────────
    meta_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Inpainting Validation Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#f8fafc;color:#1e293b;padding:32px;}}
  h1{{font-size:1.5rem;font-weight:700;color:#0f172a;margin-bottom:4px;}}
  h2{{font-size:1.1rem;font-weight:600;color:#334155;margin:28px 0 12px;
      padding-bottom:6px;border-bottom:2px solid #e2e8f0;}}
  h3{{font-size:.95rem;font-weight:600;color:#475569;margin:18px 0 8px;}}
  .subtitle{{font-size:.85rem;color:#64748b;margin-bottom:24px;}}
  .meta-table{{border-collapse:collapse;font-size:.83rem;margin-bottom:20px;}}
  .meta-table th{{text-align:left;padding:4px 12px 4px 0;color:#64748b;font-weight:500;width:140px;}}
  .meta-table td{{padding:4px 0;color:#1e293b;}}
  .original-img img{{max-width:320px;border-radius:8px;border:1px solid #e2e8f0;}}
  .variant-block{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                   padding:14px 16px;margin-bottom:14px;}}
  .variant-header{{font-size:.83rem;color:#475569;margin-bottom:10px;line-height:1.6;}}
  .triplet{{display:flex;gap:14px;}}
  .triplet-item{{flex:1;}}
  .triplet-item img{{width:100%;border-radius:6px;border:1px solid #e2e8f0;}}
  .img-lbl{{font-size:.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;
             letter-spacing:.04em;margin-bottom:4px;}}
  .badge-best{{background:#dcfce7;color:#16a34a;padding:1px 8px;border-radius:999px;
                font-size:.75rem;font-weight:600;}}
  .badge-worst{{background:#fee2e2;color:#dc2626;padding:1px 8px;border-radius:999px;
                 font-size:.75rem;font-weight:600;}}
  .metrics-table{{border-collapse:collapse;width:100%;font-size:.82rem;}}
  .metrics-table th{{background:#f1f5f9;padding:6px 10px;text-align:left;
                      border-bottom:1px solid #e2e8f0;font-weight:600;color:#475569;}}
  .metrics-table td{{padding:5px 10px;border-bottom:1px solid #f1f5f9;color:#1e293b;}}
  .metrics-table tr:hover td{{background:#f8fafc;}}
  .crop-row{{display:flex;gap:14px;margin-top:10px;padding-top:10px;border-top:1px solid #f1f5f9;}}
</style>
</head>
<body>
<h1>Inpainting Validation Report</h1>
<div class="subtitle">Generated {timestamp}</div>

<h2>Run Configuration</h2>
<table class="meta-table">{meta_rows}</table>

<h2>Original Image</h2>
<div class="original-img"><img src="{orig_uri}"></div>
<p style="font-size:.8rem;color:#94a3b8;margin-top:4px;">{out.get('sample_path','')}</p>

<h2>Variants — Masked · Inpainted · Difference</h2>
{variant_rows_html}

{chart_section}

<h2>Statistics Table</h2>
{table_html}
</body>
</html>"""
    return html


def _summary_stats_html(rows: list, use_lpips: bool) -> str:
    df = pd.DataFrame(rows)
    metrics = ["mask_area_pct", "psnr", "ssim", "l1"]
    if use_lpips and "lpips" in df.columns:
        metrics.append("lpips")
    labels = {"mask_area_pct": "Area %", "psnr": "PSNR", "ssim": "SSIM", "l1": "L1", "lpips": "LPIPS"}
    items = ""
    for metric in metrics:
        mu = df[metric].mean()
        sd = df[metric].std()
        items += (
            f"<div class='stat-item'>"
            f"<div class='stat-val'>{mu:.3f}</div>"
            f"<div class='stat-lbl'>{labels[metric]} ±{sd:.3f}</div>"
            f"</div>"
        )
    return f"<div class='summary-box'><div class='summary-row'>{items}</div></div>"


def _best_worst_html(rows: list) -> str:
    df = pd.DataFrame(rows)
    best_i = int(df["psnr"].idxmax())
    worst_i = int(df["psnr"].idxmin())
    best_v = rows[best_i]
    worst_v = rows[worst_i]
    return (
        f"<span class='badge-best'>Best: Variant {best_v['variant']} — PSNR {best_v['psnr']:.2f} dB</span>"
        f"&nbsp;&nbsp;"
        f"<span class='badge-worst'>Worst: Variant {worst_v['variant']} — PSNR {worst_v['psnr']:.2f} dB</span>"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Inpainting Validation", layout="wide")
    _inject_css()

    st.markdown("## Inpainting Validation")
    st.markdown("<div class='app-subtitle'>Single-image · multi-mask · interactive analysis</div>", unsafe_allow_html=True)

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div class='sb-section'>Dataset</div>", unsafe_allow_html=True)
        dataset_key = st.selectbox("Dataset", DATASET_OPTIONS, index=0, label_visibility="collapsed")
        st.markdown("<div class='sb-section'>Split</div>", unsafe_allow_html=True)
        split = st.selectbox("Split", ["val", "train"], index=0, label_visibility="collapsed")

        stats = _dataset_stats(dataset_key, split)
        if stats["h"] is not None:
            st.markdown(
                f"<div class='stats-line'>{stats['w']}×{stats['h']} &nbsp;·&nbsp; "
                f"train {stats['train_count']:,} &nbsp;·&nbsp; val {stats['val_count']:,}</div>",
                unsafe_allow_html=True,
            )
        elif stats["count"] == 0:
            st.markdown("<div class='ckpt-missing'>Split is empty</div>", unsafe_allow_html=True)

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='sb-section'>Model</div>", unsafe_allow_html=True)
        model_key = st.selectbox("Model", MODEL_OPTIONS, index=0, label_visibility="collapsed")
        auto_ckpt = _find_best_ckpt(dataset_key, model_key)
        if auto_ckpt is None:
            st.markdown("<div class='ckpt-missing'>✗ No checkpoint found</div>", unsafe_allow_html=True)
            ckpt_path = st.text_input("Checkpoint path", value="runs/<run>/checkpoints/best.pt")
        else:
            ckpt_path = str(auto_ckpt)
            st.markdown("<div class='ckpt-found'>✓ Checkpoint found</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ckpt-path'>{ckpt_path}</div>", unsafe_allow_html=True)

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='sb-section'>Mask</div>", unsafe_allow_html=True)
        mask_name = st.selectbox("Mask type", MASK_OPTIONS, index=0, label_visibility="collapsed")

        img_h = int(stats["h"] or 256)
        img_w = int(stats["w"] or 256)

        if mask_name in {"block", "multi_block"}:
            if "ratio_pct_text" not in st.session_state:
                st.session_state["ratio_pct_text"] = "5, 10, 20, 30"
            if "ratio_px_text" not in st.session_state:
                st.session_state["ratio_px_text"] = ", ".join(
                    f"{x:.1f}" for x in _ratios_to_pixels([5, 10, 20, 30], img_h, img_w)
                )
            if "ratio_sync_lock" not in st.session_state:
                st.session_state["ratio_sync_lock"] = False

            def _sync_px_from_pct():
                if st.session_state["ratio_sync_lock"]: return
                st.session_state["ratio_sync_lock"] = True
                try:
                    px = _ratios_to_pixels(_parse_ratios(st.session_state["ratio_pct_text"]), img_h, img_w)
                    st.session_state["ratio_px_text"] = ", ".join(f"{x:.1f}" for x in px)
                finally:
                    st.session_state["ratio_sync_lock"] = False

            def _sync_pct_from_px():
                if st.session_state["ratio_sync_lock"]: return
                st.session_state["ratio_sync_lock"] = True
                try:
                    rs = _pixels_to_ratios(_parse_pixels(st.session_state["ratio_px_text"]), img_h, img_w)
                    st.session_state["ratio_pct_text"] = ", ".join(f"{x:.1f}" for x in rs)
                finally:
                    st.session_state["ratio_sync_lock"] = False

            st.markdown("<div class='sb-section'>Ratios — side % of min(H,W)</div>", unsafe_allow_html=True)
            st.text_input("Ratios %", key="ratio_pct_text", on_change=_sync_px_from_pct, label_visibility="collapsed")
            st.markdown("<div class='sb-section'>Ratios — block side (px)</div>", unsafe_allow_html=True)
            st.text_input("Block px", key="ratio_px_text", on_change=_sync_pct_from_px, label_visibility="collapsed")

            try:
                n_masks = len(_parse_ratios(st.session_state["ratio_pct_text"]))
                st.markdown(f"<div class='stats-line'>{n_masks} mask(s)</div>", unsafe_allow_html=True)
            except Exception:
                n_masks = 0
                st.error("Invalid ratios — e.g. 5, 10, 20, 30")
            freeform_strokes = 6
        else:
            st.markdown("<div class='sb-section'>Strokes</div>", unsafe_allow_html=True)
            freeform_strokes = st.slider("Strokes", 2, 24, 6, label_visibility="collapsed")
            st.markdown("<div class='sb-section'>N masks</div>", unsafe_allow_html=True)
            n_masks = st.slider("N masks", 1, 12, 6, label_visibility="collapsed")
            if "ratio_pct_text" not in st.session_state:
                st.session_state["ratio_pct_text"] = "5, 10, 20, 30"

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='sb-section'>Metrics</div>", unsafe_allow_html=True)
        metric_scope = st.selectbox("Scope", ["mask", "full"], index=0, label_visibility="collapsed")
        use_lpips = st.checkbox("Include LPIPS", value=False)

        st.markdown("<hr class='sb-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='sb-section'>Seed</div>", unsafe_allow_html=True)
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1, label_visibility="collapsed")
        run_btn = st.button("▶  Run Inference", use_container_width=True)

    # ── main area ─────────────────────────────────────────────────────────────
    if not run_btn:
        st.info("Configure the sidebar and press **▶ Run Inference**.")
        return

    try:
        if not Path(ckpt_path).exists():
            st.error(f"Checkpoint not found: {ckpt_path}")
            return
        if n_masks <= 0:
            st.error("No valid ratios — add at least one ratio.")
            return

        _seed_everything(int(seed))
        ratios = _parse_ratios(st.session_state["ratio_pct_text"]) if mask_name in {"block", "multi_block"} else []
        area_ratios = [_ui_ratio_to_area_ratio(r, img_h, img_w) for r in ratios] if ratios else []

        out = _run_inference(
            ckpt_path=ckpt_path, dataset_key=dataset_key, split=split,
            mask_name=mask_name, ratios=area_ratios, freeform_strokes=int(freeform_strokes),
            n_masks=int(n_masks), seed=int(seed), metric_scope=metric_scope, use_lpips=bool(use_lpips),
        )
    except Exception as e:
        st.exception(e)
        return

    # summary chips
    run_meta = {
        "Dataset": dataset_key,
        "Split": split,
        "Model": model_key,
        "Mask type": mask_name,
        "Metric scope": metric_scope,
        "Seed": int(seed),
        "Sample index": out["sample_index"],
        "Image size": f"{out['image_shape'][2]}×{out['image_shape'][1]}",
        "Checkpoint": str(ckpt_path) if ckpt_path else "—",
    }
    if ratios:
        run_meta["Ratios"] = ", ".join(f"{r:.1f}%" for r in ratios)
    elif mask_name == "freeform":
        run_meta["Freeform strokes"] = freeform_strokes

    chips = [
        dataset_key, split, model_key, mask_name,
        f"scope: {metric_scope}", f"seed {int(seed)}",
        f"sample #{out['sample_index']}",
        f"{out['image_shape'][2]}×{out['image_shape'][1]}",
    ]
    if ratios:
        chips.append("ratios: " + ", ".join(f"{r:.1f}%" for r in ratios))
    elif mask_name == "freeform":
        chips.append(f"{freeform_strokes} strokes")
    st.markdown(
        "<div class='chip-row'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div>",
        unsafe_allow_html=True,
    )

    # best/worst callout
    if len(out["metrics"]) > 1:
        st.markdown(f"<div style='margin-bottom:10px;'>{_best_worst_html(out['metrics'])}</div>", unsafe_allow_html=True)

    # original
    st.markdown("<div class='sec-header'>Original</div>", unsafe_allow_html=True)
    col_img, _ = st.columns([1, 2])
    with col_img:
        st.image(out["original"], clamp=True, use_container_width=True)
        st.caption(f"Path: {out['sample_path']}")

    # pairs with diff heatmap
    st.markdown("<div class='sec-header'>Variants — Masked · Inpainted · Difference</div>", unsafe_allow_html=True)
    for i, row in enumerate(out["metrics"]):
        badge = ""
        psnr_vals = [r["psnr"] for r in out["metrics"]]
        if len(psnr_vals) > 1:
            if row["psnr"] == max(psnr_vals):
                badge = " <span class='badge-best'>best</span>"
            elif row["psnr"] == min(psnr_vals):
                badge = " <span class='badge-worst'>worst</span>"
        meta = (
            f"Variant {row['variant']}{badge} &nbsp;·&nbsp; "
            f"area {row['mask_area_pct']:.1f}% &nbsp;·&nbsp; "
            f"L1 {row['l1']:.4f} &nbsp;·&nbsp; "
            f"PSNR {row['psnr']:.2f} dB &nbsp;·&nbsp; "
            f"SSIM {row['ssim']:.4f}"
        )
        if "lpips" in row:
            meta += f" &nbsp;·&nbsp; LPIPS {row['lpips']:.4f}"
        st.markdown(f"<div class='pair-meta'>{meta}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='img-label'>Masked</div>", unsafe_allow_html=True)
            st.image(out["masked"][i], clamp=True, use_container_width=True)
        with c2:
            st.markdown("<div class='img-label'>Inpainted</div>", unsafe_allow_html=True)
            st.image(out["recon"][i], clamp=True, use_container_width=True)
        with c3:
            st.markdown("<div class='img-label'>Difference</div>", unsafe_allow_html=True)
            st.image(out["diff"][i], clamp=True, use_container_width=True)
        orig_crop, recon_crop = out["crops"][i]
        with st.expander("Zoom — masked region"):
            cz1, cz2, _ = st.columns([1, 1, 1])
            with cz1:
                st.markdown("<div class='img-label'>Crop · Original</div>", unsafe_allow_html=True)
                st.image(orig_crop, clamp=True, use_container_width=True)
            with cz2:
                st.markdown("<div class='img-label'>Crop · Inpainted</div>", unsafe_allow_html=True)
                st.image(recon_crop, clamp=True, use_container_width=True)
        st.markdown("<hr style='border-color:#f1f5f9; margin:10px 0;'>", unsafe_allow_html=True)

    # analysis plots
    st.markdown("<div class='sec-header'>Analysis</div>", unsafe_allow_html=True)

    if len(out["metrics"]) > 1:
        tab1, tab2 = st.tabs(["Metric vs Mask Area", "Per-Variant Bars"])
        with tab1:
            _chart_metric_vs_area(out["metrics"], use_lpips=bool(use_lpips))
        with tab2:
            _chart_metric_bars(out["metrics"], use_lpips=bool(use_lpips))
    else:
        st.caption("Add more than one ratio to see comparison charts.")

    # summary stats
    st.markdown(
        _summary_stats_html(out["metrics"], use_lpips=bool(use_lpips)),
        unsafe_allow_html=True,
    )

    # metrics table
    st.markdown("<div class='sec-header'>Statistics Table</div>", unsafe_allow_html=True)
    df = pd.DataFrame(out["metrics"])
    fmt = {c: "{:.4f}" for c in df.columns if c not in {"variant"}}
    fmt["mask_area_pct"] = "{:.2f}"
    fmt["psnr"] = "{:.2f}"
    st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)

    # HTML report download
    st.markdown("<div class='sec-header'>Export</div>", unsafe_allow_html=True)
    with st.spinner("Building HTML report…"):
        html_report = _generate_html_report(out, use_lpips=bool(use_lpips), meta=run_meta)
    st.download_button(
        label="⬇ Download HTML Report",
        data=html_report.encode("utf-8"),
        file_name=f"inpainting_report_{dataset_key}_{mask_name}_s{int(seed)}.html",
        mime="text/html",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()