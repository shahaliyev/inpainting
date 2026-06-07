"""
Placement distribution plots for inpainting reconstructability.

Reads per-image placement CSVs produced by eval_placement.py and draws
overlaid score distributions — one curve per mask placement — so that
the effect of spatial position on reconstruction quality is immediately
visible.

Layout
------
One figure per metric (PSNR, SSIM, LPIPS, L1).
  Rows   = visual domains  (carpet / DTD / ImageNet-simple / -complex)
  Cols   = mask severities (5 % … 40 %)
  Curves = one per placement (top-left, top-right, center, bottom-left,
           bottom-right), overlaid on the same axes.

Two curve styles are supported:
  --kde   Kernel-density estimate of the raw per-image scores (accurate,
          shows true shape).
  default Gaussian N(μ, σ) approximation from per-group mean and std
          (cleaner, faster, fine for large N).

Output
------
  figures/placement/dist_<metric>.png   one figure per metric
  figures/placement/pdf/dist_<metric>.pdf

Usage
-----
  python placement/distribution.py
  python placement/distribution.py --kde
  python placement/distribution.py --severity 10 30 --domain dtd carpet
  python placement/distribution.py --data_dir runs/placement --out_dir figures/placement
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm as scipy_norm

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Visual constants ───────────────────────────────────────────────────────────

METRICS = [
    ("psnr",  "PSNR (dB)",  True),
    ("ssim",  "SSIM",       True),
    ("lpips", "LPIPS",      False),
    ("l1",    "L1",         False),
]
METRIC_NAMES = [m[0] for m in METRICS]

PLACEMENT_ORDER = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
PLACEMENT_LABELS = {
    "top_left":     "Top-left",
    "top_right":    "Top-right",
    "center":       "Center",
    "bottom_left":  "Bottom-left",
    "bottom_right": "Bottom-right",
}
PLACEMENT_COLORS = {
    "top_left":     "#0072B2",
    "top_right":    "#D55E00",
    "center":       "#009E73",
    "bottom_left":  "#CC79A7",
    "bottom_right": "#E69F00",
}
PLACEMENT_LINESTYLES = {
    "top_left":     "-",
    "top_right":    "--",
    "center":       "-.",
    "bottom_left":  ":",
    "bottom_right": (0, (3, 1, 1, 1)),
}

DOMAIN_ORDER  = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS = {
    "carpet":           "Carpet",
    "dtd":              "DTD",
    "imagenet-simple":  "ImageNet-Simple",
    "imagenet-complex": "ImageNet-Complex",
}

# Files produced by this script — never load them back as input.
_GENERATED_STEMS = {
    "placement_summary", "placement_stats", "placement_posthoc",
    "placement_lmm_global", "placement_anova_global",
}
_REQUIRED_COLS = {
    "model", "domain", "image", "mask_severity", "mask_placement",
    "l1", "psnr", "ssim", "lpips",
}


# ── Style ──────────────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    mpl.rcParams.update({
        "font.family":      "serif",
        "mathtext.fontset": "cm",
        "font.size":         8,
        "axes.labelsize":   10,
        "axes.titlesize":    9,
        "legend.fontsize":   7,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "axes.linewidth":    0.7,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.3,
        "lines.linewidth":   1.4,
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "savefig.pad_inches": 0.02,
    })


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot per-placement score distributions from eval_placement CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data_dir", default=str(REPO_ROOT / "runs" / "placement"),
        help="Directory containing per-image placement *.csv files.",
    )
    ap.add_argument(
        "--out_dir", default=str(REPO_ROOT / "figures" / "placement"),
        help="Output directory for figures.",
    )
    ap.add_argument(
        "--kde", action="store_true",
        help="Use kernel-density estimate instead of Gaussian N(mean,std) approximation.",
    )
    ap.add_argument(
        "--severity", type=int, nargs="+", default=None,
        help="Restrict to these mask severities (e.g. --severity 10 30). "
             "Default: all found in data.",
    )
    ap.add_argument(
        "--domain", nargs="+", default=None,
        help="Restrict to these domains.",
    )
    ap.add_argument(
        "--model", nargs="+", default=None,
        help="Restrict to these model names.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate raw per-image placement CSVs.

    Skips generated output files and CSVs missing required columns.
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    skipped: list[str] = []

    for f in csv_files:
        if f.stem in _GENERATED_STEMS:
            skipped.append(f"{f.name} (generated — skipped)")
            continue

        try:
            header = pd.read_csv(f, nrows=0)
        except Exception as e:
            skipped.append(f"{f.name} (read error: {e})")
            continue

        missing = _REQUIRED_COLS - set(header.columns)
        if missing:
            skipped.append(f"{f.name} (missing: {sorted(missing)})")
            continue

        df = pd.read_csv(f)
        for col in ("l1", "psnr", "ssim", "lpips"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs.append(df)

    if skipped:
        print(f"  Skipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"    • {s}")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    if "mask_placement" in out.columns:
        present = [p for p in PLACEMENT_ORDER if p in out["mask_placement"].unique()]
        out["mask_placement"] = pd.Categorical(
            out["mask_placement"], categories=present, ordered=True
        )
    return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ordered_domains(domains) -> list[str]:
    present = set(domains)
    return [d for d in DOMAIN_ORDER if d in present] + sorted(
        d for d in present if d not in DOMAIN_ORDER
    )


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(path, dpi=dpi, **kw)
    pdf_dir = path.parent / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    fig.savefig(pdf_dir / path.with_suffix(".pdf").name, **kw)
    plt.close(fig)
    print(f"  saved {path}")


def _gaussian_curve(
    values: np.ndarray, x_grid: np.ndarray
) -> np.ndarray | None:
    """Evaluate N(μ, σ) on x_grid. Returns None if std ≈ 0."""
    mu, sigma = float(np.nanmean(values)), float(np.nanstd(values))
    if sigma < 1e-9:
        return None
    return scipy_norm.pdf(x_grid, loc=mu, scale=sigma)


def _kde_curve(
    values: np.ndarray, x_grid: np.ndarray
) -> np.ndarray | None:
    """Evaluate KDE on x_grid. Returns None if too few points."""
    clean = values[~np.isnan(values)]
    if len(clean) < 5 or np.std(clean) < 1e-9:
        return None
    try:
        kde = gaussian_kde(clean)
        return kde(x_grid)
    except Exception:
        return None


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    label: str,
    higher: bool,
    dpi: int,
    use_kde: bool,
) -> None:
    """
    Grid of distribution panels: rows = domains, cols = severities.

    Each panel overlays one curve per placement.
    """
    domains    = _ordered_domains(df["domain"].unique())
    severities = sorted(df["mask_severity"].unique())
    placements = [p for p in PLACEMENT_ORDER if p in df["mask_placement"].cat.categories]

    n_rows = len(domains)
    n_cols = len(severities)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.0),
        squeeze=False,
        sharey=False,
    )

    # Global x-range across all data for this metric (consistent x-axis).
    all_vals = df[metric].dropna().values
    if len(all_vals) == 0:
        plt.close(fig)
        return
    x_pad = (all_vals.max() - all_vals.min()) * 0.05
    x_lo  = all_vals.min() - x_pad
    x_hi  = all_vals.max() + x_pad
    x_grid = np.linspace(x_lo, x_hi, 300)

    curve_fn = _kde_curve if use_kde else _gaussian_curve

    for row_i, domain in enumerate(domains):
        dom_df = df[df["domain"] == domain]

        for col_j, severity in enumerate(severities):
            ax  = axes[row_i, col_j]
            sub = dom_df[dom_df["mask_severity"] == severity]

            any_plotted = False
            for pl in placements:
                vals = sub[sub["mask_placement"] == pl][metric].dropna().values
                if len(vals) < 5:
                    continue

                y = curve_fn(vals, x_grid)
                if y is None:
                    continue

                ax.plot(
                    x_grid, y,
                    color=PLACEMENT_COLORS[pl],
                    linestyle=PLACEMENT_LINESTYLES[pl],
                    linewidth=1.3,
                    label=PLACEMENT_LABELS[pl],
                )
                # Light fill under the curve.
                ax.fill_between(x_grid, y, alpha=0.07, color=PLACEMENT_COLORS[pl])
                any_plotted = True

            ax.set_xlim(x_lo, x_hi)
            ax.set_yticks([])
            ax.grid(True, axis="x")

            # Column header: severity (top row only).
            if row_i == 0:
                ax.set_title(f"{severity}%", fontsize=8)

            # Row label: domain (left column only).
            if col_j == 0:
                ax.set_ylabel(
                    DOMAIN_LABELS.get(domain, domain),
                    fontsize=8, labelpad=4,
                )

            if not any_plotted:
                ax.text(
                    0.5, 0.5, "no data",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="grey",
                )

    # Shared x-axis label on the bottom row only.
    for ax in axes[-1]:
        ax.set_xlabel(label, fontsize=8)

    # Single shared legend below all panels.
    handles, labels_leg = [], []
    for pl in placements:
        handles.append(
            mpl.lines.Line2D(
                [], [],
                color=PLACEMENT_COLORS[pl],
                linestyle=PLACEMENT_LINESTYLES[pl],
                linewidth=1.3,
            )
        )
        labels_leg.append(PLACEMENT_LABELS[pl])

    fig.legend(
        handles, labels_leg,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(placements),
        frameon=True, fancybox=False,
        edgecolor="lightgrey", facecolor="white", framealpha=1.0,
        handlelength=2.0, columnspacing=1.0, borderpad=0.4,
        fontsize=7,
    )

    curve_type = "KDE" if use_kde else "Gaussian N(mean, std)"
    direction  = "↑ better" if higher else "↓ better"
    fig.suptitle(
        f"{label} ({direction}) — {curve_type} distributions by placement\n"
        "rows: domain · cols: mask severity",
        fontsize=9, y=1.01,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    _save(fig, out_dir / f"dist_{metric}.png", dpi)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()

    df = load_data(args.data_dir)
    if df.empty:
        raise SystemExit(
            f"No placement CSV files found under: {args.data_dir}\n"
            "Run eval_placement.py first."
        )

    if args.domain:
        df = df[df["domain"].isin(args.domain)]
    if args.model:
        df = df[df["model"].isin(args.model)]
    if args.severity:
        df = df[df["mask_severity"].isin(args.severity)]
    if df.empty:
        raise SystemExit("No data remains after applying filters.")

    domains    = _ordered_domains(df["domain"].unique())
    severities = sorted(df["mask_severity"].unique())
    models     = sorted(df["model"].unique())

    print(
        f"Loaded {len(df):,} rows from {args.data_dir}"
        f"\n  domains   : {domains}"
        f"\n  models    : {models}"
        f"\n  severities: {severities}"
        f"\n  placements: {list(df['mask_placement'].cat.categories)}"
        f"\n  output    : {args.out_dir}"
        f"\n  curve type: {'KDE' if args.kde else 'Gaussian N(mean,std)'}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric, label, higher in METRICS:
        if metric not in df.columns or df[metric].isna().all():
            print(f"  Skipping {metric} — not present or all NaN.")
            continue
        plot_distributions(df, out, metric, label, higher, args.dpi, args.kde)

    print(f"\nDone → {out}/")


if __name__ == "__main__":
    main()
