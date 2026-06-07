"""
Cohen's d heatmap for placement pair-wise t-test results.

Reads placement_ttest.csv produced by placement/ttest.py and draws a
symmetric 5x5 Cohen's d matrix for every (domain, metric, severity) cell.

Layout
------
One figure per metric.
  Rows = visual domains
  Cols = mask severities
  Each panel = 5x5 placement-vs-placement matrix coloured by signed Cohen's d:
    blue  = placement_a better than placement_b (positive d)
    red   = placement_b better than placement_a (negative d)
    white = no difference
  Cells with p_bonferroni >= 0.05 are hatched to indicate non-significance.

Output
------
  figures/placement/cohen_<metric>.png
  figures/placement/pdf/cohen_<metric>.pdf

Usage
-----
  python placement/cohen.py
  python placement/cohen.py --severity 5 20 40
  python placement/cohen.py --data_dir figures/placement --out_dir figures/placement
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Constants ─────────────────────────────────────────────────────────────────

METRICS = [
    ("psnr",  "PSNR (dB)",  True),
    ("ssim",  "SSIM",       True),
    ("lpips", "LPIPS",      False),
    ("l1",    "L1",         False),
]
METRIC_NAMES = [m[0] for m in METRICS]

PLACEMENT_ORDER = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
PLACEMENT_SHORT = {
    "top_left":     "TL",
    "top_right":    "TR",
    "center":       "C",
    "bottom_left":  "BL",
    "bottom_right": "BR",
}

DOMAIN_ORDER = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS = {
    "carpet":           "Carpet",
    "dtd":              "DTD",
    "imagenet-simple":  "ImageNet-S",
    "imagenet-complex": "ImageNet-C",
}


# ── Style ──────────────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    mpl.rcParams.update({
        "font.family":       "serif",
        "mathtext.fontset":  "cm",
        "font.size":          8,
        "axes.labelsize":    10,
        "axes.titlesize":     8,
        "legend.fontsize":    7,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "axes.linewidth":     0.6,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cohen's d heatmap from placement paired t-test results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data_dir", default=str(REPO_ROOT / "figures" / "placement"),
        help="Directory containing placement_ttest.csv.",
    )
    ap.add_argument(
        "--out_dir", default=str(REPO_ROOT / "figures" / "placement"),
        help="Output directory for figures.",
    )
    ap.add_argument(
        "--severity", type=int, nargs="+", default=None,
        help="Restrict to these mask severities.",
    )
    ap.add_argument(
        "--domain", nargs="+", default=None,
        help="Restrict to these domains.",
    )
    ap.add_argument(
        "--clim", type=float, default=None,
        help="Symmetric colour limit for Cohen's d (e.g. --clim 1.0). "
             "Default: auto from data max |d|.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


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


def _build_matrix(
    cell_df: pd.DataFrame,
    placements: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric NxN matrices of Cohen's d and significance flags.

    d[i, j] = Cohen's d for placement_i vs placement_j
             = positive when placement_i is better (higher score / lower error)
    sig[i, j] = 1 when Bonferroni-adjusted test is significant at 0.05
    """
    n = len(placements)
    idx = {p: i for i, p in enumerate(placements)}
    d_mat  = np.zeros((n, n))
    sig_mat = np.ones((n, n), dtype=bool)   # diagonal and missing -> True (no hatch)

    for _, row in cell_df.iterrows():
        i = idx.get(row["placement_a"])
        j = idx.get(row["placement_b"])
        if i is None or j is None:
            continue
        d   = float(row["cohens_d"]) if not np.isnan(row["cohens_d"]) else 0.0
        sig = bool(row["significant_05_adj"])
        d_mat[i, j] =  d
        d_mat[j, i] = -d
        sig_mat[i, j] = sig
        sig_mat[j, i] = sig

    return d_mat, sig_mat


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_cohen(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    label: str,
    higher: bool,
    dpi: int,
    clim: float | None,
) -> None:
    """
    Grid of Cohen's d heatmaps: rows = domains, cols = severities.

    Blue = placement_a better (positive d in the direction of `higher`).
    Red  = placement_b better.
    Hatched cells are not significant after Bonferroni correction.
    """
    domains    = _ordered_domains(df["domain"].unique())
    severities = sorted(df["mask_severity"].unique())
    placements = [p for p in PLACEMENT_ORDER if p in
                  set(df["placement_a"].unique()) | set(df["placement_b"].unique())]
    n_pl = len(placements)
    short = [PLACEMENT_SHORT[p] for p in placements]

    n_rows = len(domains)
    n_cols = len(severities)

    # For metrics where higher=False (LPIPS, L1), flip sign convention so
    # blue always means "better" regardless of metric direction.
    sign = 1.0 if higher else -1.0

    # Global colour limit.
    if clim is None:
        all_d = df["cohens_d"].dropna().abs()
        clim = float(np.ceil(all_d.quantile(0.98) * 10) / 10) if len(all_d) else 1.0
        clim = max(clim, 0.1)

    cmap = "RdBu"   # blue = positive (better_a), red = negative (better_b)

    panel_size = max(1.6, n_pl * 0.36)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    for row_i, domain in enumerate(domains):
        dom_df = df[df["domain"] == domain]

        for col_j, severity in enumerate(severities):
            ax = axes[row_i, col_j]
            cell_df = dom_df[dom_df["mask_severity"] == severity]

            d_mat, sig_mat = _build_matrix(cell_df, placements)

            # Apply sign convention (blue = better).
            im = ax.imshow(
                sign * d_mat,
                cmap=cmap, vmin=-clim, vmax=clim,
                aspect="equal", interpolation="nearest",
            )

            # Hatch non-significant cells.
            for r in range(n_pl):
                for c in range(n_pl):
                    if r == c:
                        # Diagonal: grey fill.
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor="lightgrey", edgecolor="none", zorder=1,
                        ))
                        continue
                    if not sig_mat[r, c]:
                        # Non-significant: crosshatch overlay.
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor="none",
                            edgecolor="white", linewidth=0,
                            hatch="////", zorder=2,
                        ))

            # Annotate cells with Cohen's d value.
            for r in range(n_pl):
                for c in range(n_pl):
                    if r == c:
                        continue
                    val = d_mat[r, c]
                    txt = f"{val:+.2f}" if abs(val) >= 0.01 else "0"
                    brightness = (sign * val + clim) / (2 * clim)
                    txt_color = "white" if (brightness < 0.25 or brightness > 0.75) else "black"
                    ax.text(
                        c, r, txt,
                        ha="center", va="center",
                        fontsize=5.5, color=txt_color, zorder=3,
                    )

            ax.set_xticks(range(n_pl))
            ax.set_yticks(range(n_pl))
            ax.set_xticklabels(short, fontsize=6, rotation=0)
            ax.set_yticklabels(short, fontsize=6)

            # Column header (severity) — top row only.
            if row_i == 0:
                ax.set_title(f"{severity}%", fontsize=8, pad=3)

            # Row label (domain) — left column only.
            if col_j == 0:
                ax.set_ylabel(
                    DOMAIN_LABELS.get(domain, domain),
                    fontsize=8, labelpad=4,
                )

    # Shared colourbar.
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.70])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=-clim, vmax=clim),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Cohen's d  (blue = row placement better)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    direction = "up = better" if higher else "down = better"
    fig.suptitle(
        f"Cohen's d — {label} ({direction})\n"
        "rows: domain  cols: mask severity  |  hatched = not significant (Bonferroni adj.)",
        fontsize=9, y=1.02,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    _save(fig, out_dir / f"cohen_{metric}.png", dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()

    ttest_path = Path(args.data_dir) / "placement_ttest.csv"
    if not ttest_path.exists():
        raise SystemExit(
            f"placement_ttest.csv not found at: {ttest_path}\n"
            "Run  python placement/ttest.py  first."
        )

    df = pd.read_csv(ttest_path)
    df["cohens_d"] = pd.to_numeric(df["cohens_d"], errors="coerce")

    if args.domain:
        df = df[df["domain"].isin(args.domain)]
    if args.severity:
        df = df[df["mask_severity"].isin(args.severity)]
    if df.empty:
        raise SystemExit("No data remains after applying filters.")

    domains    = _ordered_domains(df["domain"].unique())
    severities = sorted(df["mask_severity"].unique())
    metrics    = [m for m in METRIC_NAMES if m in df["metric"].unique()]

    print(
        f"Loaded {len(df):,} rows from {ttest_path}"
        f"\n  domains   : {domains}"
        f"\n  severities: {severities}"
        f"\n  metrics   : {metrics}"
        f"\n  output    : {args.out_dir}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for metric, label, higher in METRICS:
        mdf = df[df["metric"] == metric]
        if mdf.empty:
            print(f"  Skipping {metric} — not in data.")
            continue
        plot_cohen(mdf, out, metric, label, higher, args.dpi, args.clim)

    print(f"\nDone -> {out}/")


if __name__ == "__main__":
    main()
