"""
Spatial placement analysis for inpainting reconstructability.

Reads all *.csv files produced by eval_placement.py from --data_dir and writes:

  effect_curves_<metric>.png   Severity × metric curves, one line per placement,
                                faceted by domain.
  heatmap_<metric>.png         Severity × placement mean-metric grid, per domain.
  placement_summary.csv        Aggregated mean ± std per (model, domain,
                                severity, placement, metric).
  placement_stats.csv          Kruskal-Wallis H-tests across placements per
                                (domain, metric, severity). Enabled by --stats.
                                Exploratory / supporting analysis only.
  placement_lmm_global.csv     Global linear mixed-model coefficients per metric,
                                pooling all domains, models, severities, and
                                placements as fixed effects with random intercept
                                per image. This is the primary statistical model
                                for RQ1–RQ3. Enabled by --lmm (slow).
  placement_anova_global.csv   Type III ANOVA table derived from the global LMM,
                                reporting factor-level F-statistics and p-values
                                (one row per fixed-effect term per metric).
                                Enabled by --lmm (produced alongside LMM).

Statistical analysis design
---------------------------
  Primary model (placement_lmm_global.csv + placement_anova_global.csv):
    score ~ C(domain) + C(mask_severity) + C(mask_placement) + C(model)
          + C(domain):C(mask_severity)
          + C(domain):C(mask_placement)
          + C(mask_severity):C(mask_placement)
          + C(model):C(mask_placement)
    groups = image  (random intercept)
    Fitted separately per metric using ML (reml=False).
    ANOVA: Type III Wald chi-square tests on fixed-effect terms, converted to
    approximate F-statistics (chi2 / df). Reports factor-level significance.

  Supporting analysis (placement_stats.csv):
    Kruskal-Wallis H-test + ε² effect size across placements per
    (domain, metric, severity). Friedman test (repeated-measures non-parametric
    ANOVA, blocked by image) + Kendall's W effect size. Enabled by --stats.

  Post-hoc pairwise tests (placement_posthoc.csv):
    Pairwise Wilcoxon signed-rank tests across all C(5,2)=10 placement pairs
    per (domain, metric, severity). Bonferroni-corrected. Effect size r via
    normal approximation. Enabled by --posthoc.

Usage
-----
  python tools/plot_placement.py

  python tools/plot_placement.py \\
      --data_dir runs/placement --out_dir figures/placement

  # Include Kruskal-Wallis significance table:
  python tools/plot_placement.py --stats

  # Fit global linear mixed models (requires statsmodels, slow for large N):
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
from itertools import combinations
from scipy.stats import friedmanchisquare, kruskal, norm as scipy_norm, wilcoxon

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
                    help="Kruskal-Wallis + Friedman tests with effect sizes.")
    ap.add_argument("--posthoc", action="store_true",
                    help="Pairwise Wilcoxon signed-rank tests (Bonferroni-corrected).")
    ap.add_argument("--lmm", action="store_true",
                    help="Fit linear mixed models (slow; requires statsmodels).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--shared_heatmap_scale", action="store_true",
        help="Use one global colour scale across all domains in the heatmap "
             "(easier cross-domain comparison). Default: per-domain scale.",
    )
    return ap.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

# Output files written by this script — must not be loaded back as input data.
_GENERATED_STEMS = {
    "placement_summary",
    "placement_stats",
    "placement_posthoc",
    "placement_lmm_global",
    "placement_anova_global",
}

_REQUIRED_COLS = {
    "model", "domain", "image", "mask_severity", "mask_placement",
    "l1", "psnr", "ssim", "lpips",
}


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate raw per-image placement CSVs from data_dir.

    Skips generated output files and any CSV missing required columns.

    Required columns: model, domain, image, mask_severity, mask_placement,
                      l1, psnr, ssim, lpips.
    """
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    skipped: list[str] = []

    for f in csv_files:
        if f.stem in _GENERATED_STEMS:
            skipped.append(f"{f.name} (generated output — skipped)")
            continue

        # Peek at headers only to validate columns cheaply.
        try:
            header = pd.read_csv(f, nrows=0)
        except Exception as e:
            skipped.append(f"{f.name} (read error: {e})")
            continue

        missing = _REQUIRED_COLS - set(header.columns)
        if missing:
            skipped.append(f"{f.name} (missing columns: {sorted(missing)})")
            continue

        df = pd.read_csv(f)
        for col in ("l1", "psnr", "ssim", "lpips"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = f.stem
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
    shared_scale: bool = False,
) -> None:
    """
    Severity × placement mean-metric heatmap, one panel per domain.

    Rows = severity levels, columns = placements.
    Green = better end of the scale, red = worse.

    Color scaling:
        shared_scale=False (default) — each domain uses its own min/max so that
            within-domain variation is most visible.
        shared_scale=True  (--shared_heatmap_scale) — one global min/max across
            all domains enables direct cross-domain comparison.
    """
    domains = _ordered_domains(df["domain"].unique())
    placements = [p for p in PLACEMENT_ORDER if p in df["mask_placement"].cat.categories]
    severities = sorted(df["mask_severity"].unique())
    n = len(domains)
    nrows, ncols = _panel_grid(n)

    # Use oriented colormap (green = better regardless of metric direction).
    cmap = "RdYlGn" if higher else "RdYlGn_r"

    # Pre-compute global colour bounds when requested.
    global_vmin = float(np.nanmin(df[metric])) if shared_scale and not df[metric].isna().all() else None
    global_vmax = float(np.nanmax(df[metric])) if shared_scale and not df[metric].isna().all() else None

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

        # Per-domain colour scale (default) or shared global scale.
        if shared_scale:
            vmin, vmax = global_vmin, global_vmax
        else:
            vmin = float(np.nanmin(sub[metric])) if not sub[metric].isna().all() else 0.0
            vmax = float(np.nanmax(sub[metric])) if not sub[metric].isna().all() else 1.0

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
    Non-parametric omnibus tests across placements per (domain, metric, severity).

    Kruskal-Wallis H-test (independent groups) with ε² effect size.
    Friedman test (repeated-measures, blocked by image) with Kendall's W effect size.

    Friedman is strictly more appropriate here because the same images appear
    at all placements (complete blocks). KW is included as a reference.
    """
    rows = []
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]
    for metric in present_metrics:
        for domain in _ordered_domains(df["domain"].unique()):
            dom_df = df[df["domain"] == domain]
            for severity in sorted(dom_df["mask_severity"].unique()):
                sev_df = dom_df[dom_df["mask_severity"] == severity]
                placements_present = [
                    p for p in PLACEMENT_ORDER
                    if p in sev_df["mask_placement"].cat.categories
                ]
                groups = [
                    sev_df[sev_df["mask_placement"] == pl][metric].dropna().values
                    for pl in placements_present
                ]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) < 2:
                    continue

                # ── Kruskal-Wallis ────────────────────────────────────────────
                try:
                    H, p_kw = kruskal(*groups)
                except Exception:
                    continue
                n_total = int(sum(len(g) for g in groups))
                k = len(groups)
                epsilon_sq = float(max(0.0, (H - k + 1) / (n_total - k))) if n_total > k else float("nan")

                # ── Friedman (repeated-measures, blocked by image × model) ────
                # index=["image", "model"] prevents aggfunc="first" from silently
                # collapsing multiple model rows for the same image.
                fr_chi2 = fr_p = kendall_w = float("nan")
                n_complete = 0
                block_cols = [c for c in ("image", "model") if c in sev_df.columns]
                if block_cols:
                    pivot = (
                        sev_df.pivot_table(
                            index=block_cols, columns="mask_placement",
                            values=metric, aggfunc="mean",
                        )
                        .reindex(columns=placements_present)
                        .dropna()
                    )
                    n_complete = len(pivot)
                    if n_complete >= k:
                        try:
                            fr_chi2, fr_p = friedmanchisquare(
                                *[pivot[p].values for p in placements_present]
                            )
                            denom = n_complete * (k - 1)
                            kendall_w = float(fr_chi2 / denom) if denom > 0 else float("nan")
                        except Exception:
                            fr_chi2 = fr_p = kendall_w = float("nan")

                rows.append({
                    "metric":           metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "domain":           domain,
                    "mask_severity":    int(severity),
                    "n_placements":     k,
                    "n_images_kw":      n_total,
                    "kruskal_H":        round(float(H), 4),
                    "kruskal_p":        float(p_kw),
                    "epsilon_sq_kw":    round(epsilon_sq, 4) if not np.isnan(epsilon_sq) else float("nan"),
                    "kw_sig_05":        bool(p_kw < 0.05),
                    "kw_sig_01":        bool(p_kw < 0.01),
                    "n_complete_blocks": n_complete,
                    "friedman_chi2":    round(float(fr_chi2), 4) if not np.isnan(fr_chi2) else float("nan"),
                    "friedman_p":       float(fr_p) if not np.isnan(fr_p) else float("nan"),
                    "kendall_w":        round(float(kendall_w), 4) if not np.isnan(kendall_w) else float("nan"),
                    "friedman_sig_05":  bool(fr_p < 0.05) if not np.isnan(fr_p) else False,
                    "friedman_sig_01":  bool(fr_p < 0.01) if not np.isnan(fr_p) else False,
                })
    return pd.DataFrame(rows)


# ── Post-hoc pairwise Wilcoxon ────────────────────────────────────────────────

def compute_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests across all C(5,2)=10 placement pairs.

    Tests each (domain, metric, severity) cell. Only images present at both
    placements are used (paired data). p-values are Bonferroni-corrected for
    10 simultaneous comparisons. Effect size r = |Z| / sqrt(N) via normal
    approximation (valid for N >= ~20).

    Output columns:
        metric, domain, mask_severity, placement_a, placement_b,
        n_paired_images, n_nonzero_diff,
        wilcoxon_W, p_raw, p_bonferroni, effect_r,
        significant_05_adj, significant_01_adj
    """
    rows = []
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]

    for metric in present_metrics:
        for domain in _ordered_domains(df["domain"].unique()):
            dom_df = df[df["domain"] == domain]
            for severity in sorted(dom_df["mask_severity"].unique()):
                sev_df = dom_df[dom_df["mask_severity"] == severity]
                placements_present = [
                    p for p in PLACEMENT_ORDER
                    if p in sev_df["mask_placement"].cat.categories
                ]
                block_cols = [c for c in ("image", "model") if c in sev_df.columns]
                if len(placements_present) < 2 or not block_cols:
                    continue

                # Pivot to align paired (image, model) subjects across placements.
                # Using index=["image", "model"] prevents aggfunc from collapsing
                # multiple model rows for the same image.
                pivot = (
                    sev_df.pivot_table(
                        index=block_cols, columns="mask_placement",
                        values=metric, aggfunc="mean",
                    )
                    .reindex(columns=placements_present)
                    .dropna()
                )
                n_subjects = len(pivot)
                if n_subjects < 10:
                    continue

                pairs = list(combinations(placements_present, 2))
                n_comparisons = len(pairs)   # 10 for all 5 placements

                for pl_a, pl_b in pairs:
                    x = pivot[pl_a].values
                    y = pivot[pl_b].values
                    diff = x - y
                    n_nonzero = int(np.sum(diff != 0))

                    if n_nonzero < 5:
                        W = p_raw = r = float("nan")
                    else:
                        try:
                            W, p_raw = wilcoxon(x, y, alternative="two-sided")
                            # Effect size r = |Z| / sqrt(n_nonzero)
                            # Z derived from p-value via normal approximation.
                            p_clamped = float(np.clip(p_raw, 1e-300, 1.0 - 1e-15))
                            z = float(scipy_norm.isf(p_clamped / 2.0))
                            r = float(z / np.sqrt(n_nonzero))
                        except Exception:
                            W = p_raw = r = float("nan")

                    p_adj = (
                        float(np.clip(float(p_raw) * n_comparisons, 0.0, 1.0))
                        if not np.isnan(p_raw) else float("nan")
                    )

                    rows.append({
                        "metric":             metric,
                        "higher_is_better":   HIGHER_IS_BETTER[metric],
                        "domain":             domain,
                        "mask_severity":      int(severity),
                        "placement_a":        pl_a,
                        "placement_b":        pl_b,
                        "n_paired_subjects":  n_subjects,
                        "n_nonzero_diff":     n_nonzero,
                        "wilcoxon_W":         round(float(W), 2) if not np.isnan(W) else float("nan"),
                        "p_raw":              float(p_raw) if not np.isnan(p_raw) else float("nan"),
                        "p_bonferroni":       p_adj,
                        "effect_r":           round(r, 4) if not np.isnan(r) else float("nan"),
                        "significant_05_adj": bool(p_adj < 0.05) if not np.isnan(p_adj) else False,
                        "significant_01_adj": bool(p_adj < 0.01) if not np.isnan(p_adj) else False,
                    })

    return pd.DataFrame(rows)


# ── Linear mixed models (global) ──────────────────────────────────────────────

def compute_lmm_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Global linear mixed model, fitted separately per metric.

    Primary statistical model for RQ1–RQ3. Pools all domains, models,
    severities, and placements into a single model per metric, treating
    domain, mask_severity, mask_placement, and model as categorical fixed
    effects with selected two-way interactions. Random intercept per image.

    Formula (per metric):
        score ~ C(domain) + C(mask_severity) + C(mask_placement) + C(model)
              + C(domain):C(mask_severity)
              + C(domain):C(mask_placement)
              + C(mask_severity):C(mask_placement)
              + C(model):C(mask_placement)

    Uses ML (reml=False) because REML is not appropriate when fixed effects
    differ across compared models.

    Slow for large datasets. Run with --lmm flag only.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError(
            "statsmodels is required for --lmm. "
            "Install it with: pip install statsmodels"
        )

    required_cols = {"domain", "mask_severity", "mask_placement", "model", "image"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns required for global LMM: {missing}")

    rows: list[dict] = []
    fitted: dict[str, object] = {}
    present_metrics = [m for m in METRIC_NAMES if m in df.columns]

    formula_template = (
        "{metric} ~ C(domain) + C(mask_severity) + C(mask_placement) + C(model)"
        " + C(domain):C(mask_severity)"
        " + C(domain):C(mask_placement)"
        " + C(mask_severity):C(mask_placement)"
        " + C(model):C(mask_placement)"
    )

    for metric in present_metrics:
        sub = df.dropna(subset=[metric, "image"]).copy()
        if len(sub) < 50:
            print(f"  [LMM] Skipping {metric}: too few rows ({len(sub)}).")
            continue

        formula = formula_template.format(metric=metric)
        print(f"  [LMM] Fitting global model for {metric}  "
              f"(n={len(sub):,}, groups={sub['image'].nunique():,}) …")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md  = smf.mixedlm(formula, data=sub, groups=sub["image"])
                mdf = md.fit(reml=False, method="lbfgs", disp=False)

            params    = mdf.fe_params
            bse       = mdf.bse
            tvals     = mdf.tvalues
            pvals     = mdf.pvalues
            ci        = mdf.conf_int()
            converged = bool(getattr(mdf, "converged", True))

            for param in params.index:
                rows.append({
                    "metric":         metric,
                    "parameter":      str(param),
                    "coef":           float(params[param]),
                    "std_err":        float(bse.get(param, float("nan"))),
                    "z":              float(tvals.get(param, float("nan"))),
                    "p_value":        float(pvals.get(param, float("nan"))),
                    "ci_lower":       float(ci.loc[param, 0]) if param in ci.index else float("nan"),
                    "ci_upper":       float(ci.loc[param, 1]) if param in ci.index else float("nan"),
                    "n_obs":          int(mdf.nobs),
                    "n_groups":       int(mdf.ngroups),
                    "log_likelihood": float(mdf.llf),
                    "converged":      converged,
                })

            fitted[metric] = mdf
            status = "converged" if converged else "DID NOT CONVERGE"
            print(f"  [LMM] {metric}: {status}  "
                  f"n={mdf.nobs:,}  groups={mdf.ngroups:,}  "
                  f"llf={mdf.llf:.2f}")

        except Exception as e:
            print(f"  [LMM] FAILED for {metric}: {e}")

    return pd.DataFrame(rows), fitted


# ── ANOVA on global LMM ───────────────────────────────────────────────────────

def compute_anova_global(
    df: pd.DataFrame,
    lmm_results: "dict[str, object]",
) -> pd.DataFrame:
    """
    Type III Wald chi-square ANOVA on the global LMM fixed effects.

    Reports factor-level significance (one row per term per metric), answering
    "is this factor as a whole significant?" rather than individual contrasts.
    Derived from the already-fitted MixedLMResults objects passed in via
    lmm_results (metric -> mdf), so no re-fitting is needed.

    Output columns:
        metric, term, df, chi2, F_approx, p_value, significant_05, significant_01
    """
    rows = []
    for metric, mdf in lmm_results.items():
        try:
            # Wald chi-square test on each fixed-effect term (robust across
            # statsmodels versions).  scalar=False gives multi-df tests when a
            # categorical factor has k>2 levels.
            wald  = mdf.wald_test_terms(scalar=False)
            n_obs = int(mdf.nobs)

            # ── Extract term rows robustly ───────────────────────────────────
            # statsmodels may return a WaldTestResults object (not a plain dict).
            # Prefer .table (DataFrame) → summary_frame() → dict iteration.
            term_rows: list[tuple[str, float, int, float]] = []

            if hasattr(wald, "table") and isinstance(wald.table, pd.DataFrame):
                tbl = wald.table
                for term in tbl.index:
                    row_ = tbl.loc[term]
                    # Column names vary: "statistic"/"chi2", "df"/"df_constraint",
                    # "P>chi2"/"pvalue" etc.
                    chi2_val = float(
                        row_.get("statistic", row_.get("chi2", row_.iloc[0]))
                    )
                    df_val = int(
                        row_.get("df", row_.get("df_constraint", 1))
                    )
                    p_val = float(
                        row_.get("P>chi2", row_.get("pvalue", row_.iloc[-1]))
                    )
                    term_rows.append((str(term), chi2_val, df_val, p_val))

            elif hasattr(wald, "summary_frame"):
                sf = wald.summary_frame()
                for term in sf.index:
                    row_ = sf.loc[term]
                    term_rows.append(
                        (str(term), float(row_.iloc[0]), int(row_.iloc[1]), float(row_.iloc[-1]))
                    )

            else:
                # Fallback: dict-like (old statsmodels behaviour).
                for term, res in wald.items():
                    chi2_val = float(res.statistic)
                    df_val   = int(getattr(res, "df_constraint", getattr(res, "df", 1)))
                    p_val    = float(res.pvalue)
                    term_rows.append((str(term), chi2_val, df_val, p_val))

            for term, chi2_val, df_val, p_val in term_rows:
                f_approx = chi2_val / df_val if df_val > 0 else float("nan")
                denom_om = chi2_val + n_obs - df_val
                omega_sq = (
                    float(max(0.0, (chi2_val - df_val) / denom_om))
                    if denom_om > 0 else float("nan")
                )
                rows.append({
                    "metric":          metric,
                    "term":            term,
                    "df":              df_val,
                    "chi2":            round(chi2_val, 4),
                    "F_approx":        round(f_approx, 4),
                    "omega_sq":        round(omega_sq, 4) if not np.isnan(omega_sq) else float("nan"),
                    "p_value":         p_val,
                    "significant_05":  bool(p_val < 0.05),
                    "significant_01":  bool(p_val < 0.01),
                })
            print(f"  [ANOVA] {metric}: {len(term_rows)} terms tested.")
        except Exception as e:
            print(f"  [ANOVA] FAILED for {metric}: {e}")

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
        plot_heatmap(df, out, metric, label, higher, args.dpi,
                     shared_scale=args.shared_heatmap_scale)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = compute_summary(df)
    summary_path = out / "placement_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  saved {summary_path}")

    # ── Kruskal-Wallis + Friedman ─────────────────────────────────────────────
    if args.stats:
        stats_df = compute_stats(df)
        stats_path = out / "placement_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  saved {stats_path}")

        if not stats_df.empty:
            kw_sig  = stats_df[stats_df["kw_sig_05"]]
            fr_sig  = stats_df[stats_df["friedman_sig_05"]]
            print(
                f"\n  Kruskal-Wallis : {len(kw_sig)}/{len(stats_df)} conditions significant at α=0.05"
                f"\n  Friedman       : {len(fr_sig)}/{len(stats_df)} conditions significant at α=0.05"
            )

    # ── Post-hoc pairwise Wilcoxon ────────────────────────────────────────────
    if args.posthoc:
        print("\n  Running pairwise Wilcoxon signed-rank tests…")
        posthoc_df = compute_posthoc(df)
        if not posthoc_df.empty:
            posthoc_path = out / "placement_posthoc.csv"
            posthoc_df.to_csv(posthoc_path, index=False)
            print(f"  saved {posthoc_path}")
            sig_pairs = posthoc_df[posthoc_df["significant_05_adj"]]
            total_pairs = len(posthoc_df)
            print(
                f"\n  Post-hoc (Bonferroni): {len(sig_pairs)}/{total_pairs} pairs "
                f"significant at α=0.05 (adjusted)"
            )
        else:
            print("  No post-hoc results produced.")

    # ── Global LMM + ANOVA (primary statistical model for RQ1–RQ3) ──────────
    if args.lmm:
        print("\n  Fitting global linear mixed models (this may take a while)…")
        lmm_df, fitted_models = compute_lmm_global(df)
        if not lmm_df.empty:
            lmm_path = out / "placement_lmm_global.csv"
            lmm_df.to_csv(lmm_path, index=False)
            print(f"  saved {lmm_path}")
        else:
            print("  No LMM results produced.")

        if fitted_models:
            print("\n  Computing Type III ANOVA on fitted models…")
            anova_df = compute_anova_global(df, fitted_models)
            if not anova_df.empty:
                anova_path = out / "placement_anova_global.csv"
                anova_df.to_csv(anova_path, index=False)
                print(f"  saved {anova_path}")
            else:
                print("  No ANOVA results produced.")

    print(f"\nDone → {out}/")


if __name__ == "__main__":
    main()