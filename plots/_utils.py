"""
Shared utilities for all plots/ scripts.

No imports from the rest of the codebase — only stdlib + numpy + matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Metrics ─────────────────────────────────────────────────────────────────
# Each entry: (key_candidates, short_title, y_axis_label, higher_is_better)
METRICS: list[tuple[list[str], str, str, bool]] = [
    (["psnr", "psnr_mask", "psnr_full"],   "PSNR",      "PSNR (dB)",   True),
    (["ssim", "ssim_mask", "ssim_full"],   "SSIM",      "SSIM",        True),
    (["lpips", "lpips_mask", "lpips_full"],"LPIPS",     "LPIPS",       False),
    (["l1", "l1_mask", "l1_full"],         "Masked L1", "Masked L1",   False),
]
METRIC_CANONICAL = [m[0][0] for m in METRICS]   # ["psnr", "ssim", "lpips", "l1"]

COLORBAR_LABELS: dict[str, str] = {
    "psnr":  "PSNR (higher is better)",
    "ssim":  "SSIM (higher is better)",
    "lpips": "LPIPS (lower is better)",
    "l1":    "Masked L1 (lower is better)",
}

# ── Domain ordering and display labels ──────────────────────────────────────
DOMAIN_ORDER  = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS: dict[str, str] = {
    "carpet":           "Carpet",
    "dtd":              "DTD",
    "imagenet-simple":  "ImageNet-Simple",
    "imagenet-complex": "ImageNet-Complex",
}

# ── Model ordering and display labels ───────────────────────────────────────
MODEL_ORDER  = ["unet", "partial_conv", "gated_conv"]
MODEL_LABELS: dict[str, str] = {
    "unet":         "U-Net",
    "partial_conv": "Partial Conv",
    "gated_conv":   "Gated Conv",
}

# ── Colorblind-safe palette (Okabe–Ito) ─────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "unet":         "#0072B2",   # blue
    "partial_conv": "#D55E00",   # vermilion
    "gated_conv":   "#009E73",   # bluish-green
}
MODEL_MARKERS: dict[str, str] = {
    "unet":         "o",
    "partial_conv": "s",
    "gated_conv":   "^",
}

# Domain colors/linestyles for dispersion plots (same Okabe–Ito extension)
DOMAIN_COLORS: dict[str, str] = {
    "carpet":           "#0072B2",
    "dtd":              "#D55E00",
    "imagenet-simple":  "#009E73",
    "imagenet-complex": "#CC79A7",
}
DOMAIN_LINESTYLES: dict[str, str] = {
    "carpet":           "-",
    "dtd":              "--",
    "imagenet-simple":  "-.",
    "imagenet-complex": ":",
}

# Maximum training severity — shown as a subtle dashed vertical line
TRAIN_SEVERITY_MAX = 30

# ── Figure sizes (inches, IEEE double-column text width ≈ 7.16 in) ──────────
FIG_SINGLE  = (3.5,  2.6)   # one journal column
FIG_DOUBLE  = (7.0,  2.6)   # two journal columns
FIG_2x2     = (7.0,  5.4)   # 2×2 panel grid
FIG_2x2_SQ  = (7.0,  6.0)   # 2×2 slightly taller (heatmaps)


# ── Paper style ───────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    """Apply publication-quality rcParams. Call once at the start of each script."""
    mpl.rcParams.update({
        "font.family":          "serif",
        "mathtext.fontset":     "cm",
        "font.size":            8,
        "axes.labelsize":       8,
        "axes.titlesize":       8,
        "legend.fontsize":      7,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "axes.linewidth":       0.7,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.3,
        "lines.linewidth":      1.4,
        "lines.markersize":     3.5,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_result(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def result_identity(result: dict) -> tuple[str, str]:
    """Return (model_name, dataset_name)."""
    return result.get("model", "unknown"), result.get("dataset", "unknown")


# ── Curve extraction ──────────────────────────────────────────────────────────

def extract_curve(
    result: dict,
    family: str = "block",
) -> list[tuple[float, dict]]:
    """
    Return [(severity_pct, metrics_dict), ...] sorted by severity.
    For freeform, severity = num_strokes.
    """
    points: list[tuple[float, dict]] = []
    for cond in result.get("conditions", []):
        mask_type = Path(cond.get("mask_yaml", "")).stem
        metrics   = cond.get("metrics") or {}
        if mask_type == family:
            if family == "freeform":
                n = (cond.get("mask_overrides") or {}).get("num_strokes")
                if n is not None:
                    points.append((float(n), metrics))
            else:
                ratios = cond.get("mask_ratios") or []
                if ratios:
                    points.append((float(ratios[0]), metrics))
    points.sort(key=lambda x: x[0])
    return points


def get_xy(
    points: list[tuple[float, dict]],
    candidates: list[str],
) -> tuple[list[float], list[float]]:
    """Extract (xs, ys) using the first available metric key."""
    key = next((c for c in candidates if any(c in m for _, m in points)), None)
    if key is None:
        return [], []
    xs, ys = [], []
    for x, m in points:
        y = m.get(key)
        if y is not None:
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


# ── Math ──────────────────────────────────────────────────────────────────────

def orient(ys: list[float], higher_better: bool) -> list[float]:
    """Orient so that higher always means better reconstruction."""
    return list(ys) if higher_better else [-y for y in ys]


def normalized_auc(
    xs: list[float],
    ys: list[float],
    higher_better: bool,
) -> Optional[float]:
    """
    R_m per the paper:
      1. Orient ys (higher = better).
      2. Normalize to [0,1] over the severity range.
      3. Return trapz / (s_max - s_min).
    Returns None for fewer than 2 points.
    """
    if len(xs) < 2:
        return None
    ys_o  = orient(ys, higher_better)
    y_min, y_max = min(ys_o), max(ys_o)
    if abs(y_max - y_min) < 1e-12:
        return 1.0
    ys_n = [(y - y_min) / (y_max - y_min) for y in ys_o]
    area = sum(
        (xs[i] - xs[i - 1]) * (ys_n[i] + ys_n[i - 1]) / 2.0
        for i in range(1, len(xs))
    )
    return float(area / (xs[-1] - xs[0]))


def model_averaged_curve(
    curves: list[tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]]:
    """
    Q_hat(d,g,s): mean across probes at each shared severity.
    curves: [(xs, ys_oriented), ...]
    """
    if not curves:
        return [], []
    sets = [set(map(float, xs)) for xs, _ in curves if xs]
    if not sets:
        return [], []
    common = sorted(sets[0].intersection(*sets[1:]))
    means  = []
    for s in common:
        vals = [y for xs, ys in curves for x, y in zip(xs, ys) if abs(float(x) - s) < 1e-9]
        means.append(float(np.mean(vals)))
    return common, means


def cross_probe_dispersion(
    curves: list[tuple[list[float], list[float]]],
) -> tuple[list[float], list[float]]:
    """
    V_hat(d,g,s): mean squared deviation from Q_hat across probes.
    curves: [(xs, ys_oriented), ...]
    """
    common_xs, q_hat = model_averaged_curve(curves)
    if not common_xs:
        return [], []
    dispersions = []
    for s, q in zip(common_xs, q_hat):
        devs = [(y - q) ** 2
                for xs, ys in curves
                for x, y in zip(xs, ys)
                if abs(float(x) - s) < 1e-9]
        dispersions.append(float(np.mean(devs)) if devs else 0.0)
    return common_xs, dispersions


# ── Save helper ───────────────────────────────────────────────────────────────

def savefig(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    """Save PNG and PDF. PDFs go into a pdf/ subfolder of out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png)
    print(f"  saved {png}")
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"
    fig.savefig(pdf)
    print(f"  saved {pdf}")
    plt.close(fig)


def add_train_boundary(ax: mpl.axes.Axes) -> None:
    """Draw the subtle dashed vertical line at the training severity maximum."""
    ax.axvline(
        x=TRAIN_SEVERITY_MAX,
        color="#aaaaaa",
        linewidth=0.7,
        linestyle="--",
        zorder=2,
    )
