"""
Plot degradation curves and robustness summaries (Section 3).

Aggregates probe models into Q_hat / V_hat, orients metrics (higher = better),
and exports normalized robustness R_m summaries plus CSV tables.

Usage:
  python tools/plot_degradation_paper.py --out_dir figures/paper

  python tools/plot_degradation_paper.py --runs_root runs --protocol degradation_v1
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
    FIG_SINGLE,
    FIG_DOUBLE,
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

MASK_STYLE = {
    "block": {
        "linestyle": "-",
        "marker": MODEL_MARKERS["unet"],
        "display": "Block",
        "color": MODEL_COLORS["unet"],
    },
    "multi_block": {
        "linestyle": "--",
        "marker": MODEL_MARKERS["partial_conv"],
        "display": "Multi-block",
        "color": MODEL_COLORS["partial_conv"],
    },
}

RATIO_GEOMETRIES = (
    "block",
    # "multi_block",
)
GEOMETRY_ORDER = RATIO_GEOMETRIES
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

    return df


def aggregate_q_v(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probe-averaged reconstructability and cross-probe dispersion.

    The input scores are first collapsed to q_hat_m(p,d,g,s). These estimates
    are then averaged over probes to obtain Q_hat_m(d,g,s). V_hat_m(d,g,s) is
    computed as the population variance across probes, matching the paper's
    definition with denominator |P|.
    """
    probe_keys = ["probe", "domain", "geometry", "metric", "severity", "severity_kind"]
    cell_keys = ["domain", "geometry", "metric", "severity", "severity_kind"]

    per_probe = (
        tidy.groupby(probe_keys, sort=False)["q_oriented"]
        .mean()
        .reset_index(name="q_hat")
    )

    qhat = (
        per_probe.groupby(cell_keys, sort=False)["q_hat"]
        .mean()
        .reset_index(name="Q_hat")
    )

    merged = per_probe.merge(qhat, on=cell_keys, how="left")

    vhat = (
        merged.groupby(cell_keys, sort=False)
        .apply(lambda g: np.mean((g["q_hat"] - g["Q_hat"]) ** 2))
        .reset_index(name="V_hat")
    )

    nprobes = (
        per_probe.groupby(cell_keys, sort=False)["q_hat"]
        .count()
        .reset_index(name="n_probes")
    )

    agg = qhat.merge(vhat, on=cell_keys, how="left").merge(
        nprobes,
        on=cell_keys,
        how="left",
    )

    agg["V_hat"] = agg["V_hat"].fillna(0.0)

    return agg


def build_dispersion_table(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Create the per-severity dispersion table used for reporting."""
    out = q_agg.copy()
    out["V_std"] = np.sqrt(out["V_hat"])

    cols = [
        "domain",
        "geometry",
        "metric",
        "severity",
        "severity_kind",
        "Q_hat",
        "V_hat",
        "V_std",
        "n_probes",
    ]

    return out[cols].sort_values(cols[:5])


def build_dispersion_summary(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-probe dispersion across severity levels."""
    rows: list[dict] = []

    for key, g in q_agg.groupby(["domain", "geometry", "metric"], sort=False):
        g = g.copy()
        g["V_std"] = np.sqrt(g["V_hat"])
        idx_max = g["V_std"].idxmax()

        rows.append({
            "domain": key[0],
            "geometry": key[1],
            "metric": key[2],
            "mean_V_std": float(g["V_std"].mean()),
            "max_V_std": float(g["V_std"].max()),
            "severity_at_max_V_std": float(g.loc[idx_max, "severity"]),
            "severity_kind_at_max": str(g.loc[idx_max, "severity_kind"]),
            "Q_hat_at_max_dispersion": float(g.loc[idx_max, "Q_hat"]),
            "n_severity_points": int(len(g)),
        })

    return pd.DataFrame(rows).sort_values(["domain", "geometry", "metric"])


def build_curves_long(tidy: pd.DataFrame, q_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Export the long-form table of probe-level and aggregated curve values.

    The table contains q_hat_m(p,d,g,s), Q_hat_m(d,g,s), V_hat_m(d,g,s),
    the mean raw metric value, and counts documenting how many observations
    contributed to each probe-level estimate.
    """
    probe_keys = ["probe", "domain", "geometry", "metric", "severity", "severity_kind"]
    cell_keys = ["domain", "geometry", "metric", "severity", "severity_kind"]

    per_probe = (
        tidy.groupby(probe_keys, sort=False)
        .agg(
            q_hat=("q_oriented", "mean"),
            q_raw_mean=("q_raw", "mean"),
            n_rows=("q_oriented", "count"),
        )
        .reset_index()
    )

    q_sub = q_agg[cell_keys + ["Q_hat", "V_hat", "n_probes"]]
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
            "q_hat",
            "Q_hat",
            "V_hat",
            "n_probes",
            "n_rows",
        ]
    ].sort_values(cell_keys + ["probe"])


def _normalize_curve(
    severities: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize a degradation curve over its evaluated severity range."""
    order = np.argsort(severities)
    x = severities[order].astype(float)
    y = values[order].astype(float)

    y_min = float(y.min())
    y_max = float(y.max())

    if y_max > y_min:
        y_bar = (y - y_min) / (y_max - y_min)
    else:
        y_bar = np.full_like(y, np.nan)

    return x, y_bar


def compute_r_m(severities: np.ndarray, q_oriented: np.ndarray) -> float:
    """Compute normalized AUC R_m from a probe-level degradation curve."""
    x, y_bar = _normalize_curve(severities, q_oriented)

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

    Repeated observations are first collapsed into q_hat_m(p,d,g,s). Each
    resulting curve is then normalized to q_bar_m over its evaluated severity
    range, and R_m is computed as normalized area under the curve. The returned
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
        tidy.groupby(probe_curve_keys, sort=False)["q_oriented"]
        .mean()
        .reset_index(name="q_hat")
    )

    rows: list[dict] = []
    group_cols = ["probe", "domain", "geometry", "metric"]

    for key, g in per_probe_curve.groupby(group_cols, sort=False):
        g = g.sort_values("severity")

        s = g["severity"].to_numpy(dtype=float)
        q = g["q_hat"].to_numpy(dtype=float)

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


def _metric_label(metric: str) -> str:
    for candidates, _, y_label, _ in METRIC_SPECS:
        if candidates[0] == metric:
            return y_label

    return metric.upper()


def _oriented_metric_label(metric: str) -> str:
    """
    Return the plotted metric label.

    PSNR/SSIM are already higher-is-better. LPIPS/L1 are negated before
    plotting, so the plotted axis uses a minus sign.
    """
    label = _metric_label(metric)

    if HIGHER_IS_BETTER[metric]:
        return label

    return f"−{label}"


def _compact_subplot_shape(n_panels: int) -> tuple[int, int]:
    """Return a compact paper-style subplot layout."""
    if n_panels <= 1:
        return 1, 1

    if n_panels <= 2:
        return 1, 2

    return 2, 2


def _compact_figsize(n_panels: int) -> tuple[float, float]:
    """
    Select figure size from plots._utils.

    Four-domain figures use FIG_2x2, which is near-square but still slightly
    wider than tall.
    """
    if n_panels <= 1:
        return FIG_SINGLE

    if n_panels <= 2:
        return FIG_DOUBLE

    return FIG_2x2


def _hide_unused_axes(axes: np.ndarray, used: int) -> None:
    """Hide unused panels if fewer domains are present."""
    for ax in axes.ravel()[used:]:
        ax.set_visible(False)


def _shared_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = [], []

    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()

        if l:
            handles, labels = h, l
            break

    if not handles:
        return

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
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


def _plot_ratio_panel(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    *,
    domain: str,
    metric: str,
    probe: str | None,
    value_col: str = "q_oriented",
    show_xlabel: bool = False,
    show_ylabel: bool = False,
) -> None:
    """Plot block and multi-block degradation curves for one compact domain panel."""
    for geometry in RATIO_GEOMETRIES:
        st = MASK_STYLE[geometry]

        sub = tidy[
            (tidy["domain"] == domain)
            & (tidy["geometry"] == geometry)
            & (tidy["metric"] == metric)
            & (tidy["severity_kind"] == "ratio")
        ].copy()

        if probe is not None:
            sub = sub[sub["probe"] == probe]

        if sub.empty:
            continue

        sub = (
            sub.groupby("severity", as_index=False)[value_col]
            .mean()
            .sort_values("severity")
        )

        ax.plot(
            sub["severity"].to_numpy(),
            sub[value_col].to_numpy(),
            linestyle=st["linestyle"],
            marker=st["marker"],
            color=st["color"],
            label=st["display"],
            zorder=3,
        )

    add_train_boundary(ax)

    ax.set_title(DOMAIN_LABELS.get(domain, domain), pad=3)

    if show_xlabel:
        ax.set_xlabel("Mask area (%)")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(_oriented_metric_label(metric))
    else:
        ax.set_ylabel("")

    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))

    # Show numeric tick labels on every subplot.
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=True,
        labelleft=True,
    )

    ax.grid(True)


def plot_qhat_curves(
    tidy: pd.DataFrame,
    q_agg: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot compact near-square probe-averaged degradation curves."""
    domains = _ordered_domains(tidy["domain"].unique())
    n_domains = len(domains)

    n_rows, n_cols = _compact_subplot_shape(n_domains)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_compact_figsize(n_domains),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    metric_label = _metric_label(metric)

    for idx, domain in enumerate(domains):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        q_row = q_agg[
            (q_agg["domain"] == domain)
            & (q_agg["metric"] == metric)
        ].copy()

        q_row = q_row.rename(columns={"Q_hat": "q_oriented"})

        _plot_ratio_panel(
            ax,
            q_row,
            domain=domain,
            metric=metric,
            probe=None,
            value_col="q_oriented",
            show_xlabel=(row == n_rows - 1),
            show_ylabel=True,
        )

    _hide_unused_axes(axes, n_domains)

    fig.suptitle(
        f"Probe-averaged {metric_label} degradation",
        y=0.995,
    )

    _shared_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _draw_qhat_metric_domain_panel(
    ax: plt.Axes,
    q_agg: pd.DataFrame,
    *,
    metric: str,
    domains: list[str],
    geometry: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Draw one Q_hat metric panel with domains as curves."""
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
            domain_sub.groupby("severity", as_index=False)["Q_hat"]
            .mean()
            .sort_values("severity")
        )

        ax.plot(
            domain_curve["severity"].to_numpy(),
            domain_curve["Q_hat"].to_numpy(),
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            label=DOMAIN_LABELS.get(domain, domain),
            zorder=3,
        )

    add_train_boundary(ax)
    ax.set_title(_metric_label(metric), pad=3)

    if show_xlabel:
        ax.set_xlabel("Mask area (%)")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(_oriented_metric_label(metric))
    else:
        ax.set_ylabel("")

    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax.grid(True)


def plot_qhat_metric_grid(
    tidy: pd.DataFrame,
    q_agg: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
    geometry: str = "block",
) -> None:
    """
    Plot one compact 2x2 metric grid for Q_hat.

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

        _draw_qhat_metric_domain_panel(
            axes[row, col],
            q_agg,
            metric=metric,
            domains=domains,
            geometry=geometry,
            show_xlabel=(row == n_rows - 1),
            show_ylabel=(col == 0),
        )

    _hide_unused_axes(axes, min(len(metrics), n_rows * n_cols))

    geometry_label = MASK_STYLE.get(geometry, {}).get("display", geometry)

    fig.suptitle(
        f"Probe-averaged degradation across metrics ({geometry_label})",
        y=0.995,
    )

    _shared_domain_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_probe_curves(
    tidy: pd.DataFrame,
    metric: str,
    probe: str,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot compact near-square probe-level degradation curves."""
    domains = _ordered_domains(tidy["domain"].unique())
    n_domains = len(domains)

    n_rows, n_cols = _compact_subplot_shape(n_domains)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_compact_figsize(n_domains),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    metric_label = _metric_label(metric)
    probe_label = MODEL_LABELS.get(probe, probe)

    sub_tidy = tidy[
        (tidy["metric"] == metric)
        & (tidy["probe"] == probe)
    ].copy()

    for idx, domain in enumerate(domains):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        _plot_ratio_panel(
            ax,
            sub_tidy,
            domain=domain,
            metric=metric,
            probe=probe,
            value_col="q_oriented",
            show_xlabel=(row == n_rows - 1),
            show_ylabel=True,
        )

    _hide_unused_axes(axes, n_domains)

    fig.suptitle(
        f"{probe_label}: {metric_label} degradation",
        y=0.995,
    )

    _shared_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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
            domain_sub.groupby("severity", as_index=False)["q_oriented"]
            .mean()
            .sort_values("severity")
        )

        ax.plot(
            domain_curve["severity"].to_numpy(),
            domain_curve["q_oriented"].to_numpy(),
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            label=DOMAIN_LABELS.get(domain, domain),
            zorder=3,
        )

    add_train_boundary(ax)
    ax.set_title(_metric_label(metric), pad=3)

    if show_xlabel:
        ax.set_xlabel("Mask area (%)")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(_oriented_metric_label(metric))
    else:
        ax.set_ylabel("")

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

    probe_label = MODEL_LABELS.get(probe, probe)
    geometry_label = MASK_STYLE.get(geometry, {}).get("display", geometry)

    fig.suptitle(
        f"{probe_label}: degradation across metrics ({geometry_label})",
        y=0.995,
    )

    _shared_domain_legend(fig, axes)

    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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

    ax.set_title(_metric_label(metric), pad=3)
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

    if show_ylabel:
        ax.set_ylabel("$R_m$")
    else:
        ax.set_ylabel("")

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

    geometry_label = MASK_STYLE.get(geometry, {}).get("display", geometry)

    fig.suptitle(
        f"Robustness $R_m$ across metrics ({geometry_label})",
        y=0.97,
    )

    _shared_probe_legend(fig, axes)

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.16,
        top=0.88,
        wspace=0.22,
        hspace=0.42,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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
            "V_hat is exported in paper_curves_long.csv and paper_dispersion.csv.",
            "Per-probe all-metric figures plot one metric per panel and one visual domain per curve.",
            "LPIPS and L1 are negated so that larger oriented values indicate better reconstruction.",
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

    p_qhat = out_dir / "degradation_Qhat_all_metrics.png"

    plot_qhat_metric_grid(
        tidy,
        q_agg,
        p_qhat,
        dpi=args.dpi,
        geometry="block",
    )

    print(f"Saved {p_qhat}")
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
