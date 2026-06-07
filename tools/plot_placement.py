"""
Spatial placement analysis for inpainting reconstructability.

Reads all *.csv files produced by eval_placement.py from --data_dir and writes:

  effect_curves_<metric>.png  Severity × metric curves, one line per placement,
                               faceted by domain.
  heatmap_<metric>.png        Severity × placement mean-metric grid, per domain.
  placement_summary.csv       Aggregated mean ± std per (model, domain,
                               severity, placement, metric).
  placement_stats.csv         Kruskal-Wallis H-tests across placements per
                               (domain, metric, severity). Enabled by --stats.
  placement_lmm.csv           Linear mixed-model coefficients per (domain, metric),
                               random intercept per image. Enabled by --lmm (slow).

Usage
-----
  python tools/plot_placement.py

  python tools/plot_placement.py \\
      --data_dir runs/placement --out_dir figures/placement

  # Include Kruskal-Wallis significance table:
  python tools/plot_placement.py --stats

  # Also fit linear mixed models (requires statsmodels, slow for large N):
  python tools/plot_placement.py --lmm
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import kruskal

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Visual constants ──────────────────────────────────────────────────────────

METRICS = [
    ("psnr",  "PSNR (dB)", True),
    ("ssim",  "SSIM",      True),
    ("lpips", "LPIPS",     False),
    ("l1",    "L1",        False),
]
METRIC_NAMES = [m[0] for m in METRICS]
HIGHER_IS_BETTER = {m[0]: m[2] for m in METRICS}

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
PLACEMENT_MARKERS = {
    "top_left": "o", "top_right": "s", "center": "^",
    "bottom_left": "D", "bottom_right": "v",
}

DOMAIN_ORDER = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS = {
    "carpet": "Carpet",
    "dtd": "DTD",
    "imagenet-simple": "ImageNet-Simple",
    "imagenet-complex": "ImageNet-Complex",
}


# ── Style ─────────────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 8, "axes.labelsize": 10, "axes.titlesize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4, "grid.alpha": 0.3,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Spatial placement analysis of inpainting reconstructability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "runs" / "placement"),
                    help="Directory containing placement *.csv files.")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "figures" / "placement"),
                    help="Directory for output figures and tables.")
    ap.add_argument("--scope", default="mask", choices=("mask", "full"),
                    help="Metric scope column to read (informational; scope is baked into CSVs).")
    ap.add_argument("--domain", default=None, nargs="+",
                    help="Restrict analysis to these domains.")
    ap.add_argument("--model", default=None, nargs="+",
                    help="Restrict analysis to these model names.")
    ap.add_argument("--stats", action="store_true",
                    help="Compute and save Kruskal-Wallis tests across placements.")
    ap.add_argument("--lmm", action="store_true",
                    help="Fit linear mixed models (slow; requires statsmodels).")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all placement CSVs from data_dir."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        for col in ("l1", "psnr", "ssim", "lpips"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = f.stem
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    # Ensure categorical ordering.
    if "mask_placement" in out.columns:
        present = [p for p in PLACEMENT_ORDER if p in out["mask_placement"].unique()]
        out["mask_placement"] = pd.Categorical(
            out["mask_placement"], categories=present, ordered=True
        )
    return out


def _ordered_domains(domains) -> list[str]:
    present = set(domains)
    return [d for d in DOMAIN_ORDER if d in present] + sorted(
        d for d in present if d not in DOMAIN_ORDER
    )


def _panel_grid(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for n panels, aiming for near-square layout."""
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


# ── Save ──────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(out_path, dpi=dpi, **kw)
    pdf_dir = out_path.parent / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    fig.savefig(pdf_dir / out_path.with_suffix(".pdf").name, **kw)
    plt.close(fig)
    print(f"  saved {out_path}")


def _shared_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = [], []
    for ax in np.array(axes).ravel():
        if not ax.get_visible():
            continue
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
    if not handles:
        return
    leg = fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(labels), 5), frameon=True, fancybox=False,
        edgecolor="lightgrey", facecolor="white", framealpha=1.0,
        handlelength=2.2, columnspacing=1.2, borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)


# ── Figure 1: Effect curves ───────────────────────────────────────────────────

def plot_effect_curves(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    label: str,
    higher: bool,
    dpi: int,
) -> None:
    """
    Per-placement severity curves, faceted by domain.

    x = mask_severity (%), y = mean metric value, one line per placement.
    """
    domains = _ordered_domains(df["domain"].unique())
    placements = [p for p in PLACEMENT_ORDER if p in df["mask_placement"].cat.categories]
    n = len(domains)
    nrows, ncols = _panel_grid(n)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.0, nrows * 2.6),
        squeeze=False,
        sharey=False,
    )
    axes_flat = axes.ravel()

    for i, domain in enumerate(domains):
        ax = axes_flat[i]
        sub = df[df["domain"] == domain]
        for pl in placements:
            curve = (
                sub[sub["mask_placement"] == pl]
                .groupby("mask_severity", observed=True)[metric]
                .mean()
                .reset_index()
                .sort_values("mask_severity")
            )
            if curve.empty or curve[metric].isna().all():
                continue
            ax.plot(
                curve["mask_severity"],
                curve[metric],
                color=PLACEMENT_COLORS[pl],
                marker=PLACEMENT_MARKERS[pl],
                linestyle="-",
                label=PLACEMENT_LABELS[pl],
            )

        ax.set_title(DOMAIN_LABELS.get(domain, domain))
        ax.set_xlabel("Mask area (%)")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(df["mask_severity"].unique())))
        ax.grid(True)
        if not higher:
            ax.invert_yaxis()

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    _shared_legend(fig, axes)
    fig.suptitle(f"{label} by Placement and Severity", fontsize=10, y=1.01)
    fig.tight_layout(rect=(0, 0.10, 1, 1.0))
    _save(fig, out_dir / f"effect_curves_{metric}.png", dpi)


# ── Figure 2: Heatmap ─────────────────────────────────────────────────────────

def plot_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    label: str,
    higher: bool,
    dpi: int,
) -> None:
    """
    Severity × placement mean-metric heatmap, one panel per domain.

    Rows = severity levels, columns = placements.
    Green = better end of the scale, red = worse.
    """
    domains = _ordered_domains(df["domain"].unique())
    placements = [p for p in PLACEMENT_ORDER if p in df["mask_placement"].cat.categories]
    severities = sorted(df["mask_severity"].unique())
    n = len(domains)
    nrows, ncols = _panel_grid(n)

    # Use oriented colormap (green = better regardless of metric direction).
    cmap = "RdYlGn" if higher else "RdYlGn_r"

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.6),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, domain in enumerate(domains):
        ax = axes_flat[i]
        sub = df[df["domain"] == domain]
        pivot = (
            sub.groupby(["mask_severity", "mask_placement"], observed=True)[metric]
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(index=severities, columns=placements)
        )
        data = pivot.values.astype(float)

        # Shared color scale across domains for comparability.
        vmin = np.nanmin(df[df["domain"] == domain][metric]) if not df[metric].isna().all() else 0
        vmax = np.nanmax(df[df["domain"] == domain][metric]) if not df[metric].isna().all() else 1

        im = ax.imshow(
            data, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(placements)))
        ax.set_xticklabels(
            [PLACEMENT_LABELS[p] for p in placements],
            rotation=40, ha="right", fontsize=7,
        )
        ax.set_yticks(range(len(severities)))
        ax.set_yticklabels([f"{s}%" for s in severities], fontsize=7)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=9)

        # Annotate cells with mean value.
        val_range = vmax - vmin if (vmax - vmin) > 0 else 1.0
        for r in range(len(severities)):
            for c in range(len(placements)):
                v = data[r, c]
                if np.isnan(v):
                    continue
                brightness = (v - vmin) / val_range  # 0 = red, 1 = green
                txt_color = "white" if (brightness < 0.25 or brightness > 0.80) else "black"
                ax.text(
                    c, r, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=6, color=txt_color,
                )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    direction = "↑ better" if higher else "↓ better"
    fig.suptitle(
        f"Mean {label} ({direction}) by Severity × Placement",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir / f"heatmap_{metric}.png", dpi)


# ── Summary table ─────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated mean ± std per (model, domain, severity, placement) for all metrics."""
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]
    group_cols = ["model", "domain", "mask_severity", "mask_placement"]

    agg = (
        df.groupby(group_cols, observed=True)[present_metrics]
        .agg(["mean", "std", "count"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ── Kruskal-Wallis tests ──────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test across placements for each (domain, metric, severity).

    Tests the null hypothesis that metric distributions are equal across all
    five spatial placements. Non-parametric; no normality assumption.
    """
    rows = []
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]
    for metric in present_metrics:
        for domain in _ordered_domains(df["domain"].unique()):
            dom_df = df[df["domain"] == domain]
            for severity in sorted(dom_df["mask_severity"].unique()):
                sev_df = dom_df[dom_df["mask_severity"] == severity]
                groups = [
                    sev_df[sev_df["mask_placement"] == pl][metric].dropna().values
                    for pl in PLACEMENT_ORDER
                    if pl in sev_df["mask_placement"].cat.categories
                ]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue
                try:
                    H, p = kruskal(*groups)
                except Exception:
                    continue
                rows.append({
                    "metric":          metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "domain":          domain,
                    "mask_severity":   int(severity),
                    "n_placements":    len(groups),
                    "n_images":        int(sum(len(g) for g in groups)),
                    "kruskal_H":       round(float(H), 4),
                    "p_value":         float(p),
                    "significant_05":  bool(p < 0.05),
                    "significant_01":  bool(p < 0.01),
                })
    return pd.DataFrame(rows)


# ── Linear mixed models ───────────────────────────────────────────────────────

def compute_lmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear mixed model: metric ~ severity * placement, random intercept per image.

    Fits separately for each (domain, metric) combination. Uses REML estimation.
    Reports fixed-effect coefficients, standard errors, z-scores, and p-values.

    Slow for large datasets. Run with --lmm flag only.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError(
            "statsmodels is required for --lmm. "
            "Install it with: pip install statsmodels"
        )

    rows = []
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]

    for metric in present_metrics:
        for domain in _ordered_domains(df["domain"].unique()):
            sub = df[df["domain"] == domain].dropna(subset=[metric, "image"]).copy()
            if len(sub) < 100:
                print(f"  Skipping LMM for {domain}/{metric}: too few rows ({len(sub)}).")
                continue

            # Treat severity and placement as categorical fixed effects.
            sub["severity_cat"]   = sub["mask_severity"].astype("category")
            sub["placement_cat"] = sub["mask_placement"].astype("category")

            formula = f"{metric} ~ C(severity_cat) * C(placement_cat)"

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    md  = smf.mixedlm(formula, data=sub, groups=sub["image"])
                    mdf = md.fit(reml=True, method="lbfgs", disp=False)

                summary_table = mdf.summary().tables[1]
                for param in summary_table.index:
                    row_data = summary_table.loc[param]
                    rows.append({
                        "metric":   metric,
                        "domain":   domain,
                        "parameter": str(param),
                        "coef":     float(row_data["Coef."]),
                        "std_err":  float(row_data["Std.Err."]),
                        "z":        float(row_data["z"]),
                        "p_value":  float(row_data["P>|z|"]),
                        "ci_lower": float(row_data["[0.025"]),
                        "ci_upper": float(row_data["0.975]"]),
                        "n_obs":    int(mdf.nobs),
                        "n_groups": int(mdf.ngroups),
                        "log_likelihood": float(mdf.llf),
                    })
                print(f"  LMM fitted: domain={domain}  metric={metric}  "
                      f"n={mdf.nobs}  groups={mdf.ngroups}")
            except Exception as e:
                print(f"  LMM failed for {domain}/{metric}: {e}")

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()

    df = load_data(args.data_dir)
    if df.empty:
        raise SystemExit(
            f"No placement CSV files found under: {args.data_dir}\n"
            "Run eval_placement.py first to generate results."
        )

    # Apply optional filters.
    if args.domain:
        df = df[df["domain"].isin(args.domain)]
    if args.model:
        df = df[df["model"].isin(args.model)]
    if df.empty:
        raise SystemExit("No data remains after applying domain/model filters.")

    domains  = _ordered_domains(df["domain"].unique())
    models   = sorted(df["model"].unique())
    n_images = df["image"].nunique() if "image" in df.columns else "?"

    print(
        f"Loaded {len(df):,} rows from {args.data_dir}"
        f"\n  domains   : {domains}"
        f"\n  models    : {models}"
        f"\n  images    : {n_images}"
        f"\n  severities: {sorted(df['mask_severity'].unique())}"
        f"\n  placements: {list(df['mask_placement'].cat.categories)}"
        f"\n  output    : {args.out_dir}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    for metric, label, higher in METRICS:
        if metric not in df.columns or df[metric].isna().all():
            print(f"  Skipping {metric} — not present or all NaN.")
            continue
        plot_effect_curves(df, out, metric, label, higher, args.dpi)
        plot_heatmap(df, out, metric, label, higher, args.dpi)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = compute_summary(df)
    summary_path = out / "placement_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  saved {summary_path}")

    # ── Kruskal-Wallis ────────────────────────────────────────────────────────
    if args.stats:
        stats_df = compute_stats(df)
        stats_path = out / "placement_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  saved {stats_path}")

        # Quick summary print.
        if not stats_df.empty:
            sig = stats_df[stats_df["significant_05"]]
            print(
                f"\n  Kruskal-Wallis: {len(sig)}/{len(stats_df)} conditions "
                f"significant at α=0.05"
            )

    # ── LMM ───────────────────────────────────────────────────────────────────
    if args.lmm:
        print("\n  Fitting linear mixed models (this may take a while)…")
        lmm_df = compute_lmm(df)
        if not lmm_df.empty:
            lmm_path = out / "placement_lmm.csv"
            lmm_df.to_csv(lmm_path, index=False)
            print(f"  saved {lmm_path}")
        else:
            print("  No LMM results produced.")

    print(f"\nDone → {out}/")


if __name__ == "__main__":
    main()
