"""
Plot all degradation curves and robustness summaries at once (Section 3).

Aggregates probe models into Q_bar / V_bar, orients metrics (higher = better),
and exports normalized robustness R_m summaries plus CSV tables.

Usage:
  python plots/plot_degradation_paper.py --out_dir figures/paper

  python plots/plot_degradation_paper.py --runs_root runs --protocol degradation_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT))

from plots._utils import (  # noqa: E402
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    DOMAIN_LINESTYLES,
    DOMAIN_ORDER,
    FIG_2x2,
    METRIC_CANONICAL,
    METRICS as METRIC_SPECS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    add_train_boundary,
    set_paper_style,
)

METRICS: tuple[str, ...] = tuple(METRIC_CANONICAL)

HIGHER_IS_BETTER: dict[str, bool] = {
    candidates[0]: higher_better
    for candidates, _, _, higher_better in METRIC_SPECS
}

SCOPES: tuple[str, ...] = ("mask", "full")

GEOMETRY_ORDER = ("block",)
PROBE_ORDER = tuple(MODEL_ORDER)
EXPECTED_N_PROBES = len(PROBE_ORDER)

DOMAIN_MARKERS: dict[str, str] = {
    "carpet": "o",
    "dtd": "s",
    "imagenet-simple": "^",
    "imagenet-complex": "D",
}

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

PROBE_SHORT_LABELS: dict[str, str] = {
    "unet": "U-Net",
    "partial_conv": "PConv",
    "gated_conv": "GConv",
}

X_LABEL_FONTSIZE = 10
VBAR_SCALE_FACTOR = 2e-5


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate degradation curves, dispersion tables, and R_m summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--runs_root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory containing training runs (eval JSONs under */eval/...)",
    )
    ap.add_argument("--protocol", default="degradation_v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--epoch", default="*", help="Epoch glob segment for discovery")
    ap.add_argument("--out_dir", default="figures/paper")
    ap.add_argument("--scope", default="mask", choices=("mask", "full"))
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def orient_value(metric: str, value: float) -> float:
    """Orient a metric value so that larger values indicate better quality."""
    if HIGHER_IS_BETTER[metric]:
        return float(value)
    return float(-value)


def _normalize_values(values: pd.Series) -> pd.Series:
    """Min-max normalize one oriented degradation curve over severity."""
    y_min = float(values.min())
    y_max = float(values.max())

    if y_max > y_min:
        return (values - y_min) / (y_max - y_min)

    return pd.Series(np.nan, index=values.index, dtype=float)


def _parse_condition(cond: dict) -> dict:
    """Pull eval_mask + intensity out of a single condition entry."""
    mask_yaml = cond.get("mask_yaml", "") or ""
    eval_mask = Path(mask_yaml).stem or "unknown"
    mask_ratios = cond.get("mask_ratios") or []

    if eval_mask in ("block", "multi_block") and mask_ratios:
        return {
            "eval_mask": eval_mask,
            "intensity": float(mask_ratios[0]),
            "intensity_kind": "ratio",
        }

    return {
        "eval_mask": eval_mask,
        "intensity": float("nan"),
        "intensity_kind": "unknown",
    }


def load_eval_file(path: str | Path) -> list[dict]:
    """Read one eval_results.json under runs/ into long-format rows."""
    path = Path(path).resolve()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    base = {
        "model": data.get("model", "unknown"),
        "dataset": data.get("dataset", "unknown"),
        "train_mask": data.get("mask", "unknown"),
        "epoch": data.get("epoch"),
        "checkpoint": data.get("checkpoint_name"),
        "source": str(path),
    }

    rows: list[dict] = []

    for cond in data.get("conditions", []):
        parsed = _parse_condition(cond)
        metrics = cond.get("metrics", {}) or {}

        for metric in METRICS:
            for scope in SCOPES:
                key = f"{metric}_{scope}"
                value = metrics.get(key)

                if value is None and scope == "mask":
                    value = metrics.get(metric)

                if value is None:
                    continue

                rows.append({
                    **base,
                    "condition": cond.get("condition"),
                    "eval_mask": parsed["eval_mask"],
                    "intensity": parsed["intensity"],
                    "intensity_kind": parsed["intensity_kind"],
                    "metric": metric,
                    "scope": scope,
                    "value": float(value),
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                })

    return rows


def load_all_evals(
    runs_root: str | Path,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
) -> pd.DataFrame:
    """Glob eval_results.json under runs_root and return a tidy DataFrame."""
    runs_root = Path(runs_root).resolve()

    pattern = str(
        runs_root / "*" / "eval" / protocol / split / f"epoch_{epoch}" / "eval_results.json"
    )

    paths = sorted(glob.glob(pattern))
    rows: list[dict] = []

    for p in paths:
        rows.extend(load_eval_file(p))

    return pd.DataFrame(rows)


def load_tidy(
    runs_root: str,
    protocol: str,
    split: str,
    epoch: str,
    scope: str,
) -> pd.DataFrame:
    df = load_all_evals(
        runs_root=runs_root,
        protocol=protocol,
        split=split,
        epoch=epoch,
    )

    if df.empty:
        return df

    df = df[df["scope"] == scope].copy()

    df = df[
        df["eval_mask"].isin(GEOMETRY_ORDER)
        & (df["intensity_kind"] == "ratio")
    ].copy()

    df["domain"] = df["dataset"]
    df["probe"] = df["model"]
    df["geometry"] = df["eval_mask"]
    df["severity"] = df["intensity"]
    df["severity_kind"] = df["intensity_kind"]
    df["q_raw"] = df["value"].astype(float)
    df["q_oriented"] = df.apply(lambda r: orient_value(r["metric"], r["q_raw"]), axis=1)
    df["q_bar"] = df.groupby(
        ["geometry", "metric", "severity_kind"],
        sort=False,
    )["q_oriented"].transform(_normalize_values)

    return df


def aggregate_q_v(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probe-averaged reconstructability and cross-probe dispersion.

    The oriented scores are first normalized into q_bar_m(p,d,g,s). These
    normalized scores are then averaged over probes to obtain Q_bar_m(d,g,s).
    V_bar_m(d,g,s) is computed as the population variance across probes,
    matching the paper's definition with denominator |P|.
    """
    probe_keys = ["probe", "domain", "geometry", "metric", "severity", "severity_kind"]
    cell_keys = ["domain", "geometry", "metric", "severity", "severity_kind"]

    per_probe = (
        tidy.groupby(probe_keys, sort=False)["q_bar"]
        .mean()
        .reset_index(name="q_bar")
    )

    qbar = (
        per_probe.groupby(cell_keys, sort=False)["q_bar"]
        .mean()
        .reset_index(name="Q_bar")
    )

    merged = per_probe.merge(qbar, on=cell_keys, how="left")

    vbar = (
        merged.groupby(cell_keys, sort=False)
        .apply(lambda g: np.mean((g["q_bar"] - g["Q_bar"]) ** 2))
        .reset_index(name="V_bar")
    )

    nprobes = (
        per_probe.groupby(cell_keys, sort=False)["q_bar"]
        .count()
        .reset_index(name="n_probes")
    )

    agg = qbar.merge(vbar, on=cell_keys, how="left").merge(
        nprobes,
        on=cell_keys,
        how="left",
    )

    agg["V_bar"] = agg["V_bar"].fillna(0.0)

    return agg


def build_dispersion_table(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Create the per-severity dispersion table used for reporting."""
    out = q_agg.copy()
    out["V_bar_std"] = np.sqrt(out["V_bar"])

    cols = [
        "domain",
        "geometry",
        "metric",
        "severity",
        "severity_kind",
        "Q_bar",
        "V_bar",
        "V_bar_std",
        "n_probes",
    ]

    return out[cols].sort_values(cols[:5])


def build_dispersion_summary(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-probe dispersion across severity levels."""
    rows: list[dict] = []

    for key, g in q_agg.groupby(["domain", "geometry", "metric"], sort=False):
        g = g.copy()
        g["V_bar_std"] = np.sqrt(g["V_bar"])
        idx_max = g["V_bar_std"].idxmax()

        rows.append({
            "domain": key[0],
            "geometry": key[1],
            "metric": key[2],
            "mean_V_bar_std": float(g["V_bar_std"].mean()),
            "max_V_bar_std": float(g["V_bar_std"].max()),
            "severity_at_max_V_bar_std": float(g.loc[idx_max, "severity"]),
            "severity_kind_at_max": str(g.loc[idx_max, "severity_kind"]),
            "Q_bar_at_max_dispersion": float(g.loc[idx_max, "Q_bar"]),
            "n_severity_points": int(len(g)),
        })

    return pd.DataFrame(rows).sort_values(["domain", "geometry", "metric"])


def build_curves_long(tidy: pd.DataFrame, q_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Export the long-form table of probe-level and aggregated curve values.

    The table contains q_bar_m(p,d,g,s), Q_bar_m(d,g,s), V_bar_m(d,g,s),
    the mean raw metric value, and counts documenting how many observations
    contributed to each probe-level estimate.
    """
    probe_keys = ["probe", "domain", "geometry", "metric", "severity", "severity_kind"]
    cell_keys = ["domain", "geometry", "metric", "severity", "severity_kind"]

    per_probe = (
        tidy.groupby(probe_keys, sort=False)
        .agg(
            q_bar=("q_bar", "mean"),
            q_oriented_mean=("q_oriented", "mean"),
            q_raw_mean=("q_raw", "mean"),
            n_rows=("q_bar", "count"),
        )
        .reset_index()
    )

    q_sub = q_agg[cell_keys + ["Q_bar", "V_bar", "n_probes"]]
    long = per_probe.merge(q_sub, on=cell_keys, how="left")

    return long[
        [
            "domain",
            "geometry",
            "metric",
            "severity",
            "severity_kind",
            "probe",
            "q_raw_mean",
            "q_oriented_mean",
            "q_bar",
            "Q_bar",
            "V_bar",
            "n_probes",
            "n_rows",
        ]
    ].sort_values(cell_keys + ["probe"])


def compute_r_m(severities: np.ndarray, q_bar: np.ndarray) -> float:
    """Compute normalized AUC R_m from a normalized probe-level degradation curve."""
    order = np.argsort(severities)
    x = severities[order].astype(float)
    y_bar = q_bar[order].astype(float)

    if len(x) < 2:
        return float("nan")

    if np.isnan(y_bar).any():
        return float("nan")

    span = float(x[-1] - x[0])

    if span <= 0:
        return float("nan")

    area = float(np.trapezoid(y_bar, x))

    return area / span


def build_robustness_table(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R_m for each probe, domain, geometry, and metric.

    R_m is computed as normalized area under the q_bar_m curve. The returned
    table also includes the mean R_m across probes for visualization.
    """
    probe_curve_keys = [
        "probe",
        "domain",
        "geometry",
        "metric",
        "severity",
        "severity_kind",
    ]

    per_probe_curve = (
        tidy.groupby(probe_curve_keys, sort=False)["q_bar"]
        .mean()
        .reset_index(name="q_bar")
    )

    rows: list[dict] = []
    group_cols = ["probe", "domain", "geometry", "metric"]

    for key, g in per_probe_curve.groupby(group_cols, sort=False):
        g = g.sort_values("severity")

        s = g["severity"].to_numpy(dtype=float)
        q = g["q_bar"].to_numpy(dtype=float)

        rec = dict(zip(group_cols, key))
        rec["R_m"] = compute_r_m(s, q)
        rec["n_points"] = int(len(s))
        rec["severity_kind"] = str(g["severity_kind"].iloc[0])

        rows.append(rec)

    per_probe = pd.DataFrame(rows)

    if per_probe.empty:
        return per_probe

    per_probe["R_m_mean_over_probes"] = per_probe.groupby(
        ["domain", "geometry", "metric"],
        sort=False,
    )["R_m"].transform("mean")

    mean_rows: list[dict] = []

    for key, g in per_probe.groupby(["domain", "geometry", "metric"], sort=False):
        mean_rows.append({
            "domain": key[0],
            "geometry": key[1],
            "metric": key[2],
            "probe": "__mean_over_probes__",
            "R_m": float(g["R_m"].mean()),
            "R_m_mean_over_probes": float(g["R_m"].mean()),
            "n_points": int(g["n_points"].max()),
            "severity_kind": str(g["severity_kind"].iloc[0]),
        })

    mean_df = pd.DataFrame(mean_rows)

    return pd.concat([per_probe, mean_df], ignore_index=True)


def _ordered_domains(domains) -> list[str]:
    present = list(domains)
    ordered = [d for d in DOMAIN_ORDER if d in present]
    ordered += sorted(d for d in present if d not in ordered)
    return ordered


def _ordered_probes(probes) -> list[str]:
    present = list(probes)
    ordered = [p for p in PROBE_ORDER if p in present]
    ordered += sorted(p for p in present if p not in ordered)
    return ordered


def _metric_subscript_label(metric: str) -> str:
    if metric == "psnr":
        return "PSNR"
    if metric == "ssim":
        return "SSIM"
    if metric == "lpips":
        return "LPIPS"
    if metric == "l1":
        return "L1"
    return metric.upper()


def _axis_metric_label(symbol: str, metric: str) -> str:
    return rf"$\bar{{{symbol}}}_{{\mathrm{{{_metric_subscript_label(metric)}}}}}$"


def _rm_axis_metric_label(metric: str) -> str:
    return rf"$R_{{\mathrm{{{_metric_subscript_label(metric)}}}}}$"


def _scale_factor_label(scale_factor: float) -> str:
    if np.isclose(scale_factor, 1e-4):
        return r"1e-4"
    if np.isclose(scale_factor, 2e-5):
        return r"2e-5"
    return f"{scale_factor:g}"


def _vbar_axis_metric_label(metric: str, scale_factor: float | None = None) -> str:
    return _axis_metric_label("V", metric)


def _save_figure(fig: plt.Figure, out_path: Path, *, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.text(
        1.035,
        0.5,
        "M",
        transform=fig.transFigure,
        fontsize=1,
        alpha=0.0,
        ha="left",
        va="center",
    )

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _hide_unused_axes(axes: np.ndarray, used: int) -> None:
    """Hide unused panels if fewer domains are present."""
    for ax in axes.ravel()[used:]:
        ax.set_visible(False)


def _draw_qbar_metric_domain_panel(
    ax: plt.Axes,
    q_agg: pd.DataFrame,
    *,
    metric: str,
    domains: list[str],
    geometry: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw one Q_bar metric panel with domains as curves."""
    sub = q_agg[
        (q_agg["metric"] == metric)
        & (q_agg["geometry"] == geometry)
        & (q_agg["severity_kind"] == "ratio")
    ].copy()

    for domain in domains:
        domain_sub = sub[sub["domain"] == domain]

        if domain_sub.empty:
            continue

        domain_curve = (
            domain_sub.groupby("severity", as_index=False)["Q_bar"]
            .mean()
            .sort_values("severity")
        )

        ax.plot(
            domain_curve["severity"].to_numpy(),
            domain_curve["Q_bar"].to_numpy(),
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            label=DOMAIN_LABELS.get(domain, domain),
            zorder=3,
        )

    add_train_boundary(ax)
    ax.set_title("")

    if show_xlabel:
        ax.set_xlabel("Mask area (%)", fontsize=X_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")

    ax.set_ylabel(_axis_metric_label("Q", metric))

    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax.grid(True)


def plot_qbar_metric_grid(
    tidy: pd.DataFrame,
    q_agg: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    geometry: str = "block",
) -> None:
    """
    Plot one compact 2x2 metric grid for Q_bar.

    Each panel is a metric; each curve is a visual domain.
    """
    domains = _ordered_domains(tidy["domain"].unique())
    metrics = [m for m in METRICS if m in tidy["metric"].values]

    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=FIG_2x2,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for idx, metric in enumerate(metrics[: n_rows * n_cols]):
        row, col = divmod(idx, n_cols)

        _draw_qbar_metric_domain_panel(
            axes[row, col],
            q_agg,
            metric=metric,
            domains=domains,
            geometry=geometry,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
        )

    _hide_unused_axes(axes, min(len(metrics), n_rows * n_cols))

    _shared_domain_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save_figure(fig, out_path, dpi=dpi)


def _draw_vbar_metric_domain_panel(
    ax: plt.Axes,
    q_agg: pd.DataFrame,
    *,
    metric: str,
    domains: list[str],
    geometry: str,
    show_xlabel: bool,
    show_ylabel: bool,
    scale_factor: float | None = None,
) -> None:
    """Draw one V_bar metric panel with domains as curves."""
    sub = q_agg[
        (q_agg["metric"] == metric)
        & (q_agg["geometry"] == geometry)
        & (q_agg["severity_kind"] == "ratio")
    ].copy()

    for domain in domains:
        domain_sub = sub[sub["domain"] == domain]

        if domain_sub.empty:
            continue

        domain_curve = (
            domain_sub.groupby("severity", as_index=False)["V_bar"]
            .mean()
            .sort_values("severity")
        )

        y = domain_curve["V_bar"].to_numpy()
        if scale_factor is not None:
            y = y / scale_factor

        ax.plot(
            domain_curve["severity"].to_numpy(),
            y,
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            label=DOMAIN_LABELS.get(domain, domain),
            zorder=3,
        )

    add_train_boundary(ax)
    ax.set_title("")

    if show_xlabel:
        ax.set_xlabel("Mask area (%)", fontsize=X_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")

    ax.set_ylabel(_vbar_axis_metric_label(metric, scale_factor))

    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    if scale_factor is None:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax.set_ylim(bottom=0)
    ax.grid(True)


def plot_vbar_metric_grid(
    tidy: pd.DataFrame,
    q_agg: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    geometry: str = "block",
    scale_factor: float | None = None,
) -> None:
    """
    Plot one compact 2x2 metric grid for V_bar.

    Each panel is a metric; each curve is a visual domain.
    """
    domains = _ordered_domains(tidy["domain"].unique())
    metrics = [m for m in METRICS if m in tidy["metric"].values]

    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=FIG_2x2,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for idx, metric in enumerate(metrics[: n_rows * n_cols]):
        row, col = divmod(idx, n_cols)

        _draw_vbar_metric_domain_panel(
            axes[row, col],
            q_agg,
            metric=metric,
            domains=domains,
            geometry=geometry,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
            scale_factor=scale_factor,
        )

    _hide_unused_axes(axes, min(len(metrics), n_rows * n_cols))

    _shared_domain_legend(fig, axes)

    if scale_factor is not None:
        fig.text(
            0.5,
            0.075,
            f"Scale factor: {_scale_factor_label(scale_factor)}",
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save_figure(fig, out_path, dpi=dpi)


def _draw_probe_metric_domain_panel(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    *,
    probe: str,
    metric: str,
    domains: list[str],
    geometry: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw one metric panel with domains as curves for a single probe."""
    sub = tidy[
        (tidy["probe"] == probe)
        & (tidy["metric"] == metric)
        & (tidy["geometry"] == geometry)
        & (tidy["severity_kind"] == "ratio")
    ].copy()

    for domain in domains:
        domain_sub = sub[sub["domain"] == domain]

        if domain_sub.empty:
            continue

        domain_curve = (
            domain_sub.groupby("severity", as_index=False)["q_bar"]
            .mean()
            .sort_values("severity")
        )

        ax.plot(
            domain_curve["severity"].to_numpy(),
            domain_curve["q_bar"].to_numpy(),
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            label=DOMAIN_LABELS.get(domain, domain),
            zorder=3,
        )

    add_train_boundary(ax)
    ax.set_title("")

    if show_xlabel:
        ax.set_xlabel("Mask area (%)", fontsize=X_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")

    ax.set_ylabel(_axis_metric_label("q", metric))

    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax.grid(True)


def _shared_domain_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    """Place a shared domain legend below a metric-grid figure."""
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
        ncol=min(len(labels), 4),
        frameon=True,
        fancybox=False,
        edgecolor="lightgrey",
        facecolor="white",
        framealpha=1.0,
        handlelength=2.2,
        columnspacing=1.2,
        borderpad=0.4,
    )

    legend.get_frame().set_linewidth(0.6)


def plot_probe_metric_grid(
    tidy: pd.DataFrame,
    probe: str,
    out_path: Path,
    *,
    dpi: int,
    geometry: str = "block",
) -> None:
    """
    Plot one compact 2x2 metric grid for a probe.

    Each panel is a metric; each curve is a visual domain.
    """
    domains = _ordered_domains(tidy["domain"].unique())
    metrics = [m for m in METRICS if m in tidy["metric"].values]

    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=FIG_2x2,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for idx, metric in enumerate(metrics[: n_rows * n_cols]):
        row, col = divmod(idx, n_cols)

        _draw_probe_metric_domain_panel(
            axes[row, col],
            tidy,
            probe=probe,
            metric=metric,
            domains=domains,
            geometry=geometry,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=True,
        )

    _hide_unused_axes(axes, min(len(metrics), n_rows * n_cols))

    _shared_domain_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save_figure(fig, out_path, dpi=dpi)


def _draw_rm_bar_panel(
    ax: plt.Axes,
    robustness: pd.DataFrame,
    *,
    metric: str,
    domains: list[str],
    probes: list[str],
    geometry: str,
    y_upper: float,
    show_xlabel: bool,
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

    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_SHORT_LABELS.get(domain, DOMAIN_LABELS.get(domain, domain)) for domain in domains],
        rotation=0,
        ha="center",
    )
    ax.set_ylim(0.0, y_upper)

    if show_xlabel:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("")

    ax.set_ylabel(_rm_axis_metric_label(metric))

    tick_step = 0.1 if y_upper <= 0.6 else 0.2
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")


def _shared_probe_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    """Place a shared probe legend below a metric-grid figure."""
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
    tidy: pd.DataFrame,
    robustness: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    geometry: str = "block",
) -> None:
    """
    Plot one compact 2x2 metric grid for robustness R_m.

    Each panel is a metric; domains are bar groups and probes are bars.
    """
    domains = _ordered_domains(tidy["domain"].unique())
    probes = _ordered_probes(tidy["probe"].unique())
    metrics = [m for m in METRICS if m in robustness["metric"].values]
    active = robustness[
        (robustness["geometry"] == geometry)
        & (robustness["probe"] != "__mean_over_probes__")
    ]
    max_rm = float(active["R_m"].max()) if not active.empty else 1.0
    y_upper = max(0.4, min(1.0, np.ceil((max_rm + 0.04) * 10.0) / 10.0))

    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_2x2[0], 4.6),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for idx, metric in enumerate(metrics[: n_rows * n_cols]):
        row, col = divmod(idx, n_cols)

        _draw_rm_bar_panel(
            axes[row, col],
            robustness,
            metric=metric,
            domains=domains,
            probes=probes,
            geometry=geometry,
            y_upper=y_upper,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
        )

    _hide_unused_axes(axes, min(len(metrics), n_rows * n_cols))

    _shared_probe_legend(fig, axes)

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.16,
        top=0.96,
        wspace=0.22,
        hspace=0.42,
    )
    _save_figure(fig, out_path, dpi=dpi)


def save_manifest(out_dir: Path, args, tidy: pd.DataFrame, n_figures: int) -> None:
    manifest = {
        "script": "plot_degradation_paper.py",
        "runs_root": str(Path(args.runs_root).resolve()),
        "protocol": args.protocol,
        "split": args.split,
        "scope": args.scope,
        "n_eval_jsons": int(tidy["source"].nunique()) if "source" in tidy.columns else None,
        "domains": sorted(tidy["domain"].unique().tolist()),
        "probes": sorted(tidy["probe"].unique().tolist()),
        "geometries": sorted(tidy["geometry"].unique().tolist()),
        "metrics": sorted(tidy["metric"].unique().tolist()),
        "n_figures": n_figures,
        "caption_notes": [
            "Block severity values are target masked-area ratios.",
            "Probe models are evaluated under the same cross-geometry protocol.",
            "Cross-probe dispersion is reported in paper_dispersion.csv and paper_dispersion_summary.csv.",
            "q_bar, Q_bar, and V_bar follow the paper notation for normalized degradation curves.",
            "Per-probe all-metric figures plot normalized q_bar_m with one metric per panel and one visual domain per curve.",
            "LPIPS and L1 are oriented before normalization so that larger normalized values indicate better reconstruction.",
        ],
    }

    path = out_dir / "paper_manifest.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {path}")


def main():
    args = parse_args()
    set_paper_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy = load_tidy(
        args.runs_root,
        args.protocol,
        args.split,
        args.epoch,
        args.scope,
    )

    if tidy.empty:
        raise SystemExit(
            f"No evaluation data under {Path(args.runs_root).resolve()!s}. "
            f"Run eval.py (protocol={args.protocol!r}, split={args.split!r}) first."
        )

    print(
        f"Loaded {tidy['source'].nunique() if 'source' in tidy.columns else '?'} runs, "
        f"{len(tidy)} rows | domains={sorted(tidy['domain'].unique())} "
        f"| probes={sorted(tidy['probe'].unique())}"
    )

    q_agg = aggregate_q_v(tidy)

    bad_probe_cells = q_agg[q_agg["n_probes"] != EXPECTED_N_PROBES]

    if not bad_probe_cells.empty:
        print(
            f"WARNING: {len(bad_probe_cells)} cells do not contain all "
            f"{EXPECTED_N_PROBES} expected probes."
        )

        print(
            bad_probe_cells[
                ["domain", "geometry", "metric", "severity", "severity_kind", "n_probes"]
            ].head(20).to_string(index=False)
        )

    curves_long = build_curves_long(tidy, q_agg)
    dispersion = build_dispersion_table(q_agg)
    dispersion_summary = build_dispersion_summary(q_agg)
    robustness = build_robustness_table(tidy)

    curves_long.to_csv(out_dir / "paper_curves_long.csv", index=False)
    dispersion.to_csv(out_dir / "paper_dispersion.csv", index=False)
    dispersion_summary.to_csv(out_dir / "paper_dispersion_summary.csv", index=False)
    robustness.to_csv(out_dir / "paper_robustness.csv", index=False)

    print(f"Saved {out_dir / 'paper_curves_long.csv'}")
    print(f"Saved {out_dir / 'paper_dispersion.csv'}")
    print(f"Saved {out_dir / 'paper_dispersion_summary.csv'}")
    print(f"Saved {out_dir / 'paper_robustness.csv'}")

    saved = 0

    p_qbar = out_dir / "degradation_Qbar_all_metrics.png"

    plot_qbar_metric_grid(
        tidy,
        q_agg,
        p_qbar,
        dpi=args.dpi,
        geometry="block",
    )

    print(f"Saved {p_qbar}")
    saved += 1

    p_vbar = out_dir / "dispersion_Vbar_all_metrics.png"

    plot_vbar_metric_grid(
        tidy,
        q_agg,
        p_vbar,
        dpi=args.dpi,
        geometry="block",
        scale_factor=VBAR_SCALE_FACTOR,
    )

    print(f"Saved {p_vbar}")
    saved += 1

    p_rm_bar = out_dir / "robustness_Rm_all_metrics_bar.png"

    plot_rm_bar_grid(
        tidy,
        robustness,
        p_rm_bar,
        dpi=args.dpi,
        geometry="block",
    )

    print(f"Saved {p_rm_bar}")
    saved += 1

    for probe in _ordered_probes(tidy["probe"].unique()):
        p_probe = out_dir / f"degradation_probe_{probe}_all_metrics.png"

        plot_probe_metric_grid(
            tidy,
            probe,
            p_probe,
            dpi=args.dpi,
            geometry="block",
        )

        print(f"Saved {p_probe}")
        saved += 1

    save_manifest(out_dir, args, tidy, saved)

    print(f"\nDone. {saved} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()
