"""
Plot paper-ready degradation curves and robustness summaries (Section 3).

Aggregates probe models into Q_hat / V_hat, orients metrics (higher = better),
and exports normalized robustness R_m heatmaps plus CSV tables.

Usage:
  python tools/plot_degradation_paper.py --out_dir figures/paper

  python tools/plot_degradation_paper.py --runs_root runs --protocol degradation_v1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

METRICS: tuple[str, ...] = ("psnr", "ssim", "lpips", "l1")
HIGHER_IS_BETTER: dict[str, bool] = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "l1": False,
}
SCOPES: tuple[str, ...] = ("mask", "full")

MASK_STYLE = {
    "block": {"linestyle": "-", "marker": "o", "display": "Block", "color": "#1f77b4"},
    "multi_block": {"linestyle": "--", "marker": "s", "display": "Multi-block", "color": "#ff7f0e"},
    "freeform": {"linestyle": "-.", "marker": "^", "display": "Freeform", "color": "#2ca02c"},
}

RATIO_GEOMETRIES = ("block", "multi_block")
DOMAIN_ORDER = ("carpet", "dtd", "imagenet-simple", "imagenet-complex")
GEOMETRY_ORDER = ("block", "multi_block", "freeform")
PROBE_ORDER = ("unet", "gated_conv", "partial_conv")
EXPECTED_N_PROBES = len(PROBE_ORDER)
PROBE_STYLE = {
    "unet": {"color": "#1f77b4", "display": "UNet"},
    "gated_conv": {"color": "#ff7f0e", "display": "Gated conv"},
    "partial_conv": {"color": "#9467bd", "display": "Partial conv"},
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
    mask_overrides = cond.get("mask_overrides") or {}

    if eval_mask in ("block", "multi_block") and mask_ratios:
        return {
            "eval_mask": eval_mask,
            "intensity": float(mask_ratios[0]),
            "intensity_kind": "ratio",
        }
    if eval_mask == "freeform" and mask_overrides.get("num_strokes") is not None:
        return {
            "eval_mask": eval_mask,
            "intensity": float(mask_overrides["num_strokes"]),
            "intensity_kind": "strokes",
        }
    return {"eval_mask": eval_mask, "intensity": float("nan"), "intensity_kind": "unknown"}


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
    df = load_all_evals(runs_root=runs_root, protocol=protocol, split=split, epoch=epoch)
    if df.empty:
        return df
    df = df[df["scope"] == scope].copy()
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

    # Estimate q_hat_m(p,d,g,s) by collapsing repeated observations.
    per_probe = (
        tidy.groupby(probe_keys, sort=False)["q_oriented"]
        .mean()
        .reset_index(name="q_hat")
    )

    # Average probe-level estimates to obtain Q_hat_m(d,g,s).
    qhat = (
        per_probe.groupby(cell_keys, sort=False)["q_hat"]
        .mean()
        .reset_index(name="Q_hat")
    )

    merged = per_probe.merge(qhat, on=cell_keys, how="left")

    # Compute V_hat_m(d,g,s) as population variance across probes.
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

    agg = qhat.merge(vhat, on=cell_keys, how="left").merge(nprobes, on=cell_keys, how="left")
    agg["V_hat"] = agg["V_hat"].fillna(0.0)

    return agg


def build_dispersion_table(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Create the per-severity dispersion table used for reporting."""
    out = q_agg.copy()
    out["V_std"] = np.sqrt(out["V_hat"])
    cols = [
        "domain", "geometry", "metric", "severity", "severity_kind",
        "Q_hat", "V_hat", "V_std", "n_probes",
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


def _normalize_curve(severities: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize a degradation curve over its evaluated severity range."""
    order = np.argsort(severities)
    x = severities[order].astype(float)
    y = values[order].astype(float)
    y_min, y_max = float(y.min()), float(y.max())
    if y_max > y_min:
        y_bar = (y - y_min) / (y_max - y_min)
    else:
        # Flat curve: min-max normalization is undefined.
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

    # Estimate q_hat_m(p,d,g,s) before computing robustness.
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

    # Add a probe-averaged robustness summary for heatmap visualization.
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
    if metric == "psnr":
        return "PSNR (dB)"
    return metric.upper()


def _panel_legend(ax: plt.Axes, *, per_panel: bool, row: int, n_rows: int, fontsize: int = 8) -> None:
    """Place legends consistently across multi-panel figures."""
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    if per_panel or row == n_rows - 1:
        ax.legend(fontsize=fontsize, loc="best")


def _plot_ratio_freeform_panels(
    axes_row: np.ndarray,
    tidy: pd.DataFrame,
    *,
    domain: str,
    metric: str,
    probe: str | None,
    metric_label: str,
    row: int,
    n_rows: int,
    value_col: str = "q_oriented",
    ylabel_suffix: str = "(oriented, ↑ better)",
    per_panel_legend: bool = False,
    geometry_colors: bool = False,
) -> None:
    """Plot block/multi-block and freeform degradation panels for one domain."""
    ax_ratio, ax_ff = axes_row[0], axes_row[1]
    legend_fs = 7 if per_panel_legend else 8

    for geometry in RATIO_GEOMETRIES:
        st = MASK_STYLE[geometry]
        sub = tidy[
            (tidy["domain"] == domain)
            & (tidy["geometry"] == geometry)
            & (tidy["metric"] == metric)
            & (tidy["severity_kind"] == "ratio")
        ]
        if probe is not None:
            sub = sub[sub["probe"] == probe]
        sub = sub.sort_values("severity")
        if sub.empty:
            continue
        plot_kw: dict = {
            "linestyle": st["linestyle"],
            "marker": st["marker"],
            "linewidth": 2.0,
            "markersize": 5,
            "label": st["display"],
        }
        if geometry_colors:
            plot_kw["color"] = st["color"]
        ax_ratio.plot(
            sub["severity"].to_numpy(),
            sub[value_col].to_numpy(),
            **plot_kw,
        )

    ax_ratio.set_ylabel(f"{domain}\n{metric_label}\n{ylabel_suffix}", fontsize=9)
    ax_ratio.set_xlabel("Mask area (%)", fontsize=9)
    ax_ratio.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax_ratio.grid(True, alpha=0.3)
    if row == 0:
        ax_ratio.set_title("Block / Multi-block", fontsize=10)
    _panel_legend(ax_ratio, per_panel=per_panel_legend, row=row, n_rows=n_rows, fontsize=legend_fs)

    st = MASK_STYLE["freeform"]
    sub = tidy[
        (tidy["domain"] == domain)
        & (tidy["geometry"] == "freeform")
        & (tidy["metric"] == metric)
        & (tidy["severity_kind"] == "strokes")
    ]
    if probe is not None:
        sub = sub[sub["probe"] == probe]
    sub = sub.sort_values("severity")
    if not sub.empty:
        ff_color = st["color"] if geometry_colors else "#2ca02c"
        ax_ff.plot(
            sub["severity"].to_numpy(),
            sub[value_col].to_numpy(),
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=5,
            color=ff_color,
            label=st["display"],
        )
    else:
        ax_ff.set_visible(False)

    ax_ff.set_xlabel("Number of strokes", fontsize=9)
    ax_ff.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax_ff.grid(True, alpha=0.3)
    if row == 0:
        ax_ff.set_title("Freeform", fontsize=10)
    if not sub.empty:
        _panel_legend(ax_ff, per_panel=per_panel_legend, row=row, n_rows=n_rows, fontsize=legend_fs)


def plot_qhat_curves(
    tidy: pd.DataFrame,
    q_agg: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot probe-averaged degradation curves Q_hat_m(d,g,s)."""
    domains = _ordered_domains(tidy["domain"].unique())
    n_rows = len(domains)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(10.5, 2.8 * n_rows),
        squeeze=False,
    )

    metric_label = _metric_label(metric)

    for row, domain in enumerate(domains):
        q_row = q_agg[
            (q_agg["domain"] == domain)
            & (q_agg["metric"] == metric)
        ].copy()

        # Reuse the shared panel renderer with Q_hat as the plotted value.
        q_row = q_row.rename(columns={"Q_hat": "q_oriented"})

        _plot_ratio_freeform_panels(
            axes[row],
            q_row,
            domain=domain,
            metric=metric,
            probe=None,
            metric_label=metric_label,
            row=row,
            n_rows=n_rows,
            value_col="q_oriented",
            ylabel_suffix=r"($\hat{Q}_m$, ↑ better)",
            per_panel_legend=False,
            geometry_colors=False,
        )

    fig.suptitle(
        f"Probe-averaged degradation  $\\hat{{Q}}_m(d,g,s)$  —  {metric_label}",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_probe_curves(
    tidy: pd.DataFrame,
    metric: str,
    probe: str,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot probe-level oriented degradation curves."""
    domains = _ordered_domains(tidy["domain"].unique())
    n_rows = len(domains)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(10.5, 2.8 * n_rows),
        squeeze=False,
    )

    metric_label = _metric_label(metric)
    probe_label = PROBE_STYLE.get(probe, {}).get("display", probe)

    sub_tidy = tidy[(tidy["metric"] == metric) & (tidy["probe"] == probe)].copy()

    for row, domain in enumerate(domains):
        _plot_ratio_freeform_panels(
            axes[row],
            sub_tidy,
            domain=domain,
            metric=metric,
            probe=probe,
            metric_label=metric_label,
            row=row,
            n_rows=n_rows,
            value_col="q_oriented",
            ylabel_suffix=r"($\tilde{q}_m$, ↑ better)",
            per_panel_legend=True,
            geometry_colors=True,
        )

    fig.suptitle(
        f"Probe degradation  $\\tilde{{q}}_m(p,d,g,s)$  —  {probe_label}  |  {metric_label}",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rm_heatmap(
    robustness: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot mean robustness R_m across probes for each domain and geometry."""
    sub = robustness[
        (robustness["metric"] == metric)
        & (robustness["probe"] != "__mean_over_probes__")
    ]
    if sub.empty:
        return
    mean_r = (
        sub.groupby(["domain", "geometry"], sort=False)["R_m"]
        .mean()
        .reset_index()
    )
    domains = _ordered_domains(mean_r["domain"].unique())
    geometries = [g for g in GEOMETRY_ORDER if g in mean_r["geometry"].unique()]
    geometries += sorted(g for g in mean_r["geometry"].unique() if g not in geometries)

    mat = np.full((len(domains), len(geometries)), np.nan)
    for i, d in enumerate(domains):
        for j, g in enumerate(geometries):
            v = mean_r[(mean_r["domain"] == d) & (mean_r["geometry"] == g)]["R_m"]
            if len(v):
                mat[i, j] = float(v.iloc[0])

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(geometries)))
    ax.set_xticklabels([MASK_STYLE.get(g, {}).get("display", g) for g in geometries], fontsize=9)
    ax.set_yticks(np.arange(len(domains)))
    ax.set_yticklabels(domains, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title(f"Robustness $R_m$ — {metric.upper()} (mean over probes)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="$R_m$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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
            "Freeform severity is represented by the configured stroke count.",
            "Block and multi-block severity values are target masked-area ratios.",
            "Probe models are evaluated under the same cross-geometry protocol.",
            "Cross-probe dispersion is reported in paper_dispersion.csv and paper_dispersion_summary.csv.",
            "V_hat is exported in paper_curves_long.csv and paper_dispersion.csv.",
            "Per-probe figures plot tilde q_m(p,d,g,s) (metrics oriented so larger is better).",
            "LPIPS and L1 are negated so that larger oriented values indicate better reconstruction.",
        ],
    }
    path = out_dir / "paper_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {path}")


def main():
    args = parse_args()
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
    for metric in METRICS:
        if metric not in tidy["metric"].values:
            print(f"Skipping {metric} — not in data")
            continue
        p_curve = out_dir / f"degradation_Qhat_{metric}.png"
        plot_qhat_curves(
            tidy, q_agg, metric, p_curve,
            dpi=args.dpi,
        )
        print(f"Saved {p_curve}")
        saved += 1

        p_heat = out_dir / f"robustness_Rm_{metric}.png"
        plot_rm_heatmap(robustness, metric, p_heat, dpi=args.dpi)
        print(f"Saved {p_heat}")
        saved += 1

        for probe in _ordered_probes(tidy["probe"].unique()):
            p_probe = out_dir / f"degradation_probe_{probe}_{metric}.png"
            plot_probe_curves(tidy, metric, probe, p_probe, dpi=args.dpi)
            print(f"Saved {p_probe}")
            saved += 1

    save_manifest(out_dir, args, tidy, saved)
    print(f"\nDone. {saved} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()