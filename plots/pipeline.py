"""
Pipeline overview figure — Figure 1 for the paper.

Draws a clean two-row schematic of the reconstructability analysis framework:

  Row 1:  Visual domains  →  Severity-controlled masks  →  Shared image-mask bank
  Row 2:  Reconstruction probes  →  Metrics  →  Analysis outputs  →  Regimes

Optionally loads real domain thumbnails from --image_dir (four images named
carpet.*, dtd.*, imagenet_simple.*, imagenet_complex.*). Falls back to
procedurally generated texture placeholders if no images are provided.

Produces (in figures/main_candidates/):
  pipeline_overview.png
  pdf/pipeline_overview.pdf

Usage:
  python plots/pipeline.py --out figures
  python plots/pipeline.py --out figures --image_dir path/to/thumbnails
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import savefig, set_paper_style


# ── Palette ───────────────────────────────────────────────────────────────────

C_BOX      = "#f0f0f0"   # light gray box fill
C_BOX_EMPh = "#e8eef6"   # slightly blue-tinted for analysis outputs
C_BOX_EDGE = "#aaaaaa"   # box border
C_ARROW    = "#888888"   # arrow color
C_TITLE    = "#222222"   # main label color
C_SUB      = "#555555"   # subtitle / annotation color

# Domain accent colors (Okabe–Ito subset, desaturated)
DOMAIN_COLORS = {
    "Carpet":            "#b3d0e8",
    "DTD":               "#f5c6a0",
    "ImageNet-Simple":   "#b8ddb8",
    "ImageNet-Complex":  "#e0c8e0",
}

MODEL_COLORS = {
    "U-Net":         "#d0e0f0",
    "Partial Conv":  "#fde8d0",
    "Gated Conv":    "#d0edd0",
}

METRIC_COLOR = "#e8e8e8"
ANALYSIS_COLOR = C_BOX_EMPh
REGIME_COLOR   = "#dde8d8"

FS_LABEL  = 7.5   # section label font size
FS_ITEM   = 6.5   # item / sub-text font size
FS_TITLE  = 8.5   # figure super-title


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, *, fc=C_BOX, ec=C_BOX_EDGE, lw=0.6, radius=0.012,
         zorder=2):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)


def _text(ax, x, y, s, *, ha="center", va="center", fs=FS_ITEM,
          color=C_TITLE, bold=False, style="normal"):
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, fontstyle=style,
            transform=ax.transAxes, clip_on=False)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.18,head_length=0.008",
            color=C_ARROW, lw=0.8,
        ),
        annotation_clip=False,
    )


def _section_label(ax, x, y_top, label):
    _text(ax, x, y_top + 0.025, label, fs=FS_LABEL, bold=True, color=C_TITLE)


def _texture_carpet(ax, rect):
    """Grid-stripe placeholder for Carpet."""
    inset = ax.inset_axes(rect)
    img = np.zeros((20, 20, 3))
    img[::4, :] = [0.7, 0.75, 0.85]
    img[:, ::4] = [0.75, 0.7, 0.85]
    inset.imshow(img, aspect="auto", interpolation="nearest")
    inset.axis("off")
    return inset


def _texture_dtd(ax, rect):
    """Random-noise placeholder for DTD (diverse texture)."""
    rng = np.random.default_rng(1)
    inset = ax.inset_axes(rect)
    img = rng.uniform(0.45, 0.85, (20, 20, 3))
    inset.imshow(img, aspect="auto", interpolation="nearest")
    inset.axis("off")
    return inset


def _texture_imagenet_simple(ax, rect):
    """Smooth gradient placeholder for ImageNet-Simple."""
    inset = ax.inset_axes(rect)
    g = np.linspace(0.4, 0.9, 20)
    img = np.stack([
        np.outer(g, np.ones(20)) * 0.55,
        np.outer(g, np.ones(20)) * 0.75,
        np.outer(g, np.ones(20)) * 0.55,
    ], axis=2)
    inset.imshow(img, aspect="auto", interpolation="bilinear")
    inset.axis("off")
    return inset


def _texture_imagenet_complex(ax, rect):
    """Multi-frequency noise placeholder for ImageNet-Complex."""
    rng = np.random.default_rng(7)
    inset = ax.inset_axes(rect)
    base = rng.uniform(0.3, 0.9, (20, 20, 3))
    # Add coarse structure
    coarse = np.kron(rng.uniform(0.0, 0.3, (5, 5, 3)), np.ones((4, 4, 1)))
    img = np.clip(base + coarse, 0, 1)
    inset.imshow(img, aspect="auto", interpolation="nearest")
    inset.axis("off")
    return inset


TEXTURE_FNS = [
    _texture_carpet,
    _texture_dtd,
    _texture_imagenet_simple,
    _texture_imagenet_complex,
]

DOMAIN_NAMES = ["Carpet", "DTD", "ImageNet-\nSimple", "ImageNet-\nComplex"]


def _load_thumbnail(ax, rect, path: Path):
    """Load an actual image file into an inset axes."""
    try:
        import PIL.Image
        img = np.asarray(PIL.Image.open(path).convert("RGB").resize((40, 40)))
        inset = ax.inset_axes(rect)
        inset.imshow(img, aspect="auto")
        inset.axis("off")
        return inset
    except Exception:
        return None


def _mask_icon(ax, x, y, w, h, ratio, *, zorder=3):
    """Draw a small image placeholder with a central masked block."""
    _box(ax, x, y, w, h, fc="#e8e8e8", ec=C_BOX_EDGE, lw=0.5)
    mw = w * ratio
    mh = h * ratio
    mx = x + (w - mw) / 2
    my = y + (h - mh) / 2
    mask_box = mpatches.FancyBboxPatch(
        (mx, my), mw, mh,
        boxstyle="round,pad=0,rounding_size=0.005",
        fc="#333333", ec="none", lw=0, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(mask_box)


def _mini_degradation_curve(ax, rect):
    """Draw a small schematic degradation curve in an inset."""
    inset = ax.inset_axes(rect)
    xs = np.linspace(0, 1, 30)
    ys = 0.9 * np.exp(-2.2 * xs) + 0.05
    inset.plot(xs, ys, color="#0072B2", lw=1.2)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#aaaaaa")
    return inset


def _mini_dispersion_icon(ax, rect):
    """Draw three model curves spreading apart as a dispersion icon."""
    inset = ax.inset_axes(rect)
    xs = np.linspace(0, 1, 30)
    offsets = [0.0, 0.12, -0.1]
    colors  = ["#0072B2", "#D55E00", "#009E73"]
    for off, col in zip(offsets, colors):
        ys = 0.5 + off * xs ** 0.7
        inset.plot(xs, ys, color=col, lw=0.9)
    inset.set_xlim(0, 1)
    inset.set_ylim(0.1, 0.9)
    inset.set_xticks([])
    inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#aaaaaa")
    return inset


# ── Main figure ───────────────────────────────────────────────────────────────

def draw_pipeline(image_dir: Path | None) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Super-title ──────────────────────────────────────────────────────────
    _text(ax, 0.5, 0.97,
          "Controlled Reconstructability Analysis",
          fs=FS_TITLE, bold=True, color="#111111")

    # ── Layout constants ─────────────────────────────────────────────────────
    # Row tops (in axes fraction, y increases upward)
    row1_top  = 0.865   # top row: domains / masks / shared bank
    row2_top  = 0.40    # bottom row: probes / metrics / analysis / regimes
    row_h     = 0.22    # box height
    thumb_h   = 0.13    # thumbnail height within domain boxes

    # Row 1 column positions and widths
    # [domains] → [masks] → [shared bank]
    r1_x  = [0.01, 0.365, 0.64]
    r1_w  = [0.32, 0.23,  0.33]

    # Row 2 column positions and widths
    # [probes] → [metrics] → [analysis] → [regimes]
    r2_x  = [0.01, 0.295, 0.515, 0.755]
    r2_w  = [0.25, 0.185, 0.21,  0.225]

    # ── ROW 1: Visual domains ────────────────────────────────────────────────
    _section_label(ax, r1_x[0] + r1_w[0] / 2, row1_top, "Visual Domains")
    _box(ax, r1_x[0], row1_top - row_h, r1_w[0], row_h)

    domain_thumb_w = 0.065
    thumb_gap = (r1_w[0] - 4 * domain_thumb_w) / 5
    for i, (name, tfn) in enumerate(zip(DOMAIN_NAMES, TEXTURE_FNS)):
        tx = r1_x[0] + thumb_gap + i * (domain_thumb_w + thumb_gap)
        ty = row1_top - row_h + 0.075

        # Try loading real image; fall back to procedural texture
        loaded = False
        if image_dir is not None:
            slug = name.replace("\n", "_").replace("-", "_").lower()
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = image_dir / f"{slug}.{ext}"
                if p.exists():
                    _load_thumbnail(ax, [tx, ty, domain_thumb_w, thumb_h], p)
                    loaded = True
                    break
        if not loaded:
            tfn(ax, [tx, ty, domain_thumb_w, thumb_h])

        _text(ax, tx + domain_thumb_w / 2, row1_top - row_h + 0.03,
              name, fs=5.5, color=C_SUB)

    # ── ROW 1: Severity-controlled masks ─────────────────────────────────────
    _section_label(ax, r1_x[1] + r1_w[1] / 2, row1_top,
                   "Severity-Controlled Masks")
    _box(ax, r1_x[1], row1_top - row_h, r1_w[1], row_h)

    severities = [0.05, 0.12, 0.22, 0.33, 0.45]
    sev_labels = ["2%", "10%", "20%", "30%", "40%"]
    n_sev = len(severities)
    icon_w = 0.030
    icon_h = 0.090
    sev_gap = (r1_w[1] - n_sev * icon_w) / (n_sev + 1)
    for i, (ratio, lbl) in enumerate(zip(severities, sev_labels)):
        ix = r1_x[1] + sev_gap + i * (icon_w + sev_gap)
        iy = row1_top - row_h + 0.07
        _mask_icon(ax, ix, iy, icon_w, icon_h, ratio)
        _text(ax, ix + icon_w / 2, row1_top - row_h + 0.03,
              lbl, fs=5.5, color=C_SUB)

    # ── ROW 1: Shared image-mask bank ────────────────────────────────────────
    _section_label(ax, r1_x[2] + r1_w[2] / 2, row1_top, "Shared Image-Mask Bank")
    _box(ax, r1_x[2], row1_top - row_h, r1_w[2], row_h)
    cx2 = r1_x[2] + r1_w[2] / 2
    _text(ax, cx2, row1_top - row_h * 0.38,
          "same images · same masks · same severities",
          fs=FS_ITEM - 0.5, color=C_SUB, style="italic")
    _text(ax, cx2, row1_top - row_h * 0.62,
          "All probes evaluated on\nidentical image-mask pairs",
          fs=FS_ITEM, color=C_TITLE)

    # ── Row 1 arrows ─────────────────────────────────────────────────────────
    mid1 = row1_top - row_h / 2
    _arrow(ax, r1_x[0] + r1_w[0], mid1, r1_x[1], mid1)
    _arrow(ax, r1_x[1] + r1_w[1], mid1, r1_x[2], mid1)

    # ── ROW 2: Reconstruction probes ─────────────────────────────────────────
    _section_label(ax, r2_x[0] + r2_w[0] / 2, row2_top, "Reconstruction Probes")
    _box(ax, r2_x[0], row2_top - row_h, r2_w[0], row_h)

    models = list(MODEL_COLORS.keys())
    model_h = 0.05
    model_gap = (row_h - len(models) * model_h) / (len(models) + 1)
    for j, m in enumerate(models):
        my = row2_top - row_h + model_gap + j * (model_h + model_gap)
        mx = r2_x[0] + 0.015
        mw = r2_w[0] - 0.030
        _box(ax, mx, my, mw, model_h, fc=MODEL_COLORS[m], ec=C_BOX_EDGE, lw=0.5)
        _text(ax, mx + mw / 2, my + model_h / 2, m, fs=FS_ITEM, color=C_TITLE)

    _text(ax, r2_x[0] + r2_w[0] / 2, row2_top - row_h - 0.025,
          "probes, not contributions", fs=5.5, color=C_SUB, style="italic")

    # ── ROW 2: Metrics ───────────────────────────────────────────────────────
    _section_label(ax, r2_x[1] + r2_w[1] / 2, row2_top, "Metrics")
    _box(ax, r2_x[1], row2_top - row_h, r2_w[1], row_h)

    metrics = ["PSNR", "SSIM", "LPIPS", "Masked L1"]
    met_h   = 0.038
    met_gap = (row_h - len(metrics) * met_h) / (len(metrics) + 1)
    for j, m in enumerate(metrics):
        my = row2_top - row_h + met_gap + j * (met_h + met_gap)
        mx = r2_x[1] + 0.012
        mw = r2_w[1] - 0.024
        _box(ax, mx, my, mw, met_h, fc=METRIC_COLOR, ec=C_BOX_EDGE, lw=0.5)
        _text(ax, mx + mw / 2, my + met_h / 2, m, fs=FS_ITEM - 0.5,
              color=C_TITLE)

    _text(ax, r2_x[1] + r2_w[1] / 2, row2_top - row_h - 0.025,
          "masked-region evaluation", fs=5.5, color=C_SUB, style="italic")

    # ── ROW 2: Analysis outputs ───────────────────────────────────────────────
    _section_label(ax, r2_x[2] + r2_w[2] / 2, row2_top, "Analysis")
    _box(ax, r2_x[2], row2_top - row_h, r2_w[2], row_h, fc=ANALYSIS_COLOR)

    # Mini degradation curve
    icon_bw = 0.075
    icon_bh = 0.07
    icon_bx = r2_x[2] + 0.015
    icon_by = row2_top - row_h + 0.115
    _mini_degradation_curve(ax, [icon_bx, icon_by, icon_bw, icon_bh])
    _text(ax, icon_bx + icon_bw / 2, row2_top - row_h + 0.09,
          "Degradation\ncurves", fs=FS_ITEM - 0.5, color=C_TITLE)

    # Mini dispersion icon
    icon_dx = r2_x[2] + r2_w[2] - icon_bw - 0.015
    _mini_dispersion_icon(ax, [icon_dx, icon_by, icon_bw, icon_bh])
    _text(ax, icon_dx + icon_bw / 2, row2_top - row_h + 0.09,
          "Cross-probe\ndispersion", fs=FS_ITEM - 0.5, color=C_TITLE)

    # ── ROW 2: Regimes ───────────────────────────────────────────────────────
    _section_label(ax, r2_x[3] + r2_w[3] / 2, row2_top,
                   "Reconstructability Regimes")
    _box(ax, r2_x[3], row2_top - row_h, r2_w[3], row_h, fc=REGIME_COLOR)

    bullets = [
        "domain-driven\ndegradation",
        "architecture-\nsensitive regimes",
        "metric-dependent\nreliability",
    ]
    bul_h   = 0.05
    bul_gap = (row_h - len(bullets) * bul_h) / (len(bullets) + 1)
    for j, b in enumerate(bullets):
        by_ = row2_top - row_h + bul_gap + j * (bul_h + bul_gap)
        _text(ax, r2_x[3] + 0.018, by_ + bul_h / 2,
              f"▸ {b}", ha="left", fs=FS_ITEM - 0.5, color=C_TITLE)

    # ── Row 2 arrows ─────────────────────────────────────────────────────────
    mid2 = row2_top - row_h / 2
    _arrow(ax, r2_x[0] + r2_w[0], mid2, r2_x[1], mid2)
    _arrow(ax, r2_x[1] + r2_w[1], mid2, r2_x[2], mid2)
    _arrow(ax, r2_x[2] + r2_w[2], mid2, r2_x[3], mid2)

    # ── Vertical connecting arrow (row 1 shared bank → row 2 probes) ─────────
    # Drop from middle of shared bank down to top of probes block
    shared_cx = r1_x[2] + r1_w[2] / 2
    probes_cx  = r2_x[0] + r2_w[0] / 2
    bridge_y_top = row1_top - row_h
    bridge_y_bot = row2_top
    # Elbow: right edge of row1 shared bank → down → left to probes
    _arrow(ax, shared_cx, bridge_y_top - 0.005,
           probes_cx,   bridge_y_bot + 0.005)

    # ── Row labels (italic, below each row) ──────────────────────────────────
    row1_label_y = row1_top - row_h - 0.04
    row2_label_y = row2_top - row_h - 0.04

    return fig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate pipeline overview figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", default="figures",
                    help="Base output directory.")
    ap.add_argument("--image_dir", default=None,
                    help="Optional directory with domain thumbnail images "
                         "(carpet.png, dtd.png, imagenet_simple.png, "
                         "imagenet_complex.png).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()

    image_dir = Path(args.image_dir) if args.image_dir else None
    fig = draw_pipeline(image_dir)

    out_dir = Path(args.out) / "main_candidates"
    savefig(fig, out_dir, "pipeline_overview")
    print("\nDone.")


if __name__ == "__main__":
    main()
