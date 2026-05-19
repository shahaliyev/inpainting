"""
Robustness R_m summary plot.

Produces (in figures/robustness/):
  robustness_Rm_all_metrics_bar.png

Usage:
  python plots/robustness.py --out figures
  python plots/robustness.py --results $all --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (  # noqa: E402
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    FIG_2x2,
    METRIC_CANONICAL,
    METRICS,
    MODEL_COLORS,
    MODEL_LABELS,
    build_robustness_table,
    load_tidy_evals,
    ordered_domains,
    ordered_models,
    savefig,
    set_paper_style,
)
from plots.plot_degradation_paper import plot_rm_bar_grid as plot_paper_rm_bar_grid

DOMAIN_SHORT_LABELS: dict[str, str] = {
    "carpet": "Carpet",
    "dtd": "DTD",
    "imagenet-simple": "IN-Simple",
    "imagenet-complex": "IN-Complex",
}

BAR_MODEL_COLORS: dict[str, str] = {
    "unet": MODEL_COLORS["unet"],
    "partial_conv": DOMAIN_COLORS["imagenet-complex"],
    "gated_conv": MODEL_COLORS["gated_conv"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Robustness R_m grouped-bar plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="Optional eval_results.json paths. If omitted, discover runs automatically.",
    )
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--protocol", default="degradation_v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--epoch", default="*", help="Epoch glob segment for discovery.")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--scope", default="mask", choices=("mask", "full"))
    ap.add_argument("--geometry", default="block")
    return ap.parse_args()


def _metric_label(metric: str) -> str:
    for candidates, _, y_label, _ in METRICS:
        if candidates[0] == metric:
            return y_label
    return metric.upper()


def _draw_rm_bar_panel(
    ax: plt.Axes,
    robustness,
    *,
    metric: str,
    domains: list[str],
    probes: list[str],
    geometry: str,
    y_upper: float,
    show_ylabel: bool,
) -> None:
    """Draw one robustness grouped-bar panel for a metric."""
    sub = robustness[
        (robustness["metric"] == metric)
        & (robustness["geometry"] == geometry)
        & (robustness["probe"] != "__mean_over_probes__")
    ].copy()

    x = np.arange(len(domains), dtype=float)
    width = 0.72 / max(len(probes), 1)
    offsets = (np.arange(len(probes)) - (len(probes) - 1) / 2.0) * width

    for offset, probe in zip(offsets, probes):
        vals = []
        for domain in domains:
            v = sub[
                (sub["domain"] == domain)
                & (sub["probe"] == probe)
            ]["R_m"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)

        ax.bar(
            x + offset,
            vals,
            width=width,
            color=BAR_MODEL_COLORS.get(probe, MODEL_COLORS.get(probe, "#333333")),
            label=MODEL_LABELS.get(probe, probe),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_title(_metric_label(metric), pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_SHORT_LABELS.get(domain, DOMAIN_LABELS.get(domain, domain)) for domain in domains],
        rotation=0,
        ha="center",
    )
    ax.set_ylim(0.0, y_upper)
    ax.set_xlabel("")
    ax.set_ylabel("$R_m$" if show_ylabel else "")
    tick_step = 0.1 if y_upper <= 0.6 else 0.2
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")


def _shared_probe_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = [], []
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    if not handles:
        return

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 3),
        frameon=True,
        fancybox=False,
        edgecolor="lightgrey",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.8,
        columnspacing=1.2,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.6)


def plot_rm_bar_grid(
    tidy,
    robustness,
    out_dir: Path,
    *,
    geometry: str = "block",
) -> None:
    """Plot one compact 2x2 metric grid for robustness R_m."""
    domains = ordered_domains(tidy["domain"].unique())
    probes = ordered_models(tidy["probe"].unique())
    metrics = [m for m in METRIC_CANONICAL if m in robustness["metric"].values]
    active = robustness[
        (robustness["geometry"] == geometry)
        & (robustness["probe"] != "__mean_over_probes__")
    ]
    max_rm = float(active["R_m"].max()) if not active.empty else 1.0
    y_upper = max(0.4, min(1.0, np.ceil((max_rm + 0.04) * 10.0) / 10.0))

    fig, axes = plt.subplots(2, 2, figsize=(FIG_2x2[0], 4.6), squeeze=False)
    for idx, metric in enumerate(metrics[:4]):
        row, col = divmod(idx, 2)
        _draw_rm_bar_panel(
            axes[row, col],
            robustness,
            metric=metric,
            domains=domains,
            probes=probes,
            geometry=geometry,
            y_upper=y_upper,
            show_ylabel=(col == 0),
        )

    for ax in axes.ravel()[min(len(metrics), 4):]:
        ax.set_visible(False)

    geometry_label = "Block" if geometry == "block" else geometry
    fig.suptitle(f"Robustness $R_m$ across metrics ({geometry_label})", y=0.97)
    _shared_probe_legend(fig, axes)
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.16,
        top=0.88,
        wspace=0.22,
        hspace=0.42,
    )
    savefig(fig, out_dir, "robustness_Rm_all_metrics_bar")


def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir = Path(args.out) / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy = load_tidy_evals(
        args.results,
        runs_root=args.runs_root,
        protocol=args.protocol,
        split=args.split,
        epoch=args.epoch,
        scope=args.scope,
        geometries=(args.geometry,),
    )
    if tidy.empty:
        raise SystemExit("No evaluation data found.")

    robustness = build_robustness_table(tidy)
    if args.geometry == "block":
        plot_paper_rm_bar_grid(
            tidy,
            robustness,
            out_dir / "robustness_Rm_all_metrics_bar.png",
            dpi=300,
            geometry="block",
        )
        print(f"  saved {out_dir / 'robustness_Rm_all_metrics_bar.png'}")
    else:
        plot_rm_bar_grid(tidy, robustness, out_dir, geometry=args.geometry)
    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
