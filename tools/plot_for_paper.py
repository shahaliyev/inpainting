"""
Generate the two paper figures: degradation curves and cross-probe dispersion.

Outputs (PNG + PDF) in --out_dir:
  degradation_all_metrics.png / pdf/degradation_all_metrics.pdf
  dispersion_all_metrics.png  / pdf/dispersion_all_metrics.pdf

Usage:
  python tools/plot_for_paper.py --out_dir figures/paper
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Constants (inlined from plots/_utils.py) ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

METRICS_SPEC = [
    (["psnr", "psnr_mask", "psnr_full"],    "PSNR",  True),
    (["ssim", "ssim_mask", "ssim_full"],    "SSIM",  True),
    (["lpips", "lpips_mask", "lpips_full"], "LPIPS", False),
    (["l1", "l1_mask", "l1_full"],          "L1",    False),
]
METRICS: tuple[str, ...] = tuple(c[0][0] for c in METRICS_SPEC)
HIGHER_IS_BETTER: dict[str, bool] = {c[0][0]: c[2] for c in METRICS_SPEC}

DOMAIN_ORDER = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS: dict[str, str] = {
    "carpet": "Carpet", "dtd": "DTD",
    "imagenet-simple": "ImageNet-Simple", "imagenet-complex": "ImageNet-Complex",
}
DOMAIN_COLORS: dict[str, str] = {
    "carpet": "#0072B2", "dtd": "#D55E00",
    "imagenet-simple": "#009E73", "imagenet-complex": "#CC79A7",
}
DOMAIN_LINESTYLES: dict[str, str] = {
    "carpet": "-", "dtd": "--", "imagenet-simple": "-.", "imagenet-complex": ":",
}
DOMAIN_MARKERS: dict[str, str] = {
    "carpet": "o", "dtd": "s", "imagenet-simple": "^", "imagenet-complex": "D",
}

MODEL_ORDER = ["unet", "partial_conv", "gated_conv"]
TRAIN_SEVERITY_MAX = 30
FIG_2x2 = (7.0, 5.4)

GEOMETRY_ORDER = ("block",)
PROBE_ORDER = tuple(MODEL_ORDER)
EXPECTED_N_PROBES = len(PROBE_ORDER)


def set_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 8, "axes.labelsize": 12, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4, "grid.alpha": 0.3,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def add_train_boundary(ax: plt.Axes) -> None:
    ax.axvline(x=TRAIN_SEVERITY_MAX, color="#aaaaaa", linewidth=0.7, linestyle="--", zorder=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs_root", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--protocol", default="degradation_v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--epoch", default="*")
    ap.add_argument("--out_dir", default="figures/paper")
    ap.add_argument("--scope", default="mask", choices=("mask", "full"))
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = {
        "model": data.get("model", "unknown"),
        "dataset": data.get("dataset", "unknown"),
        "source": str(path),
    }
    rows: list[dict] = []
    for cond in data.get("conditions", []):
        mask_yaml = cond.get("mask_yaml", "") or ""
        eval_mask = Path(mask_yaml).stem or "unknown"
        ratios = cond.get("mask_ratios") or []
        if eval_mask not in ("block", "multi_block") or not ratios:
            continue
        intensity = float(ratios[0])
        metrics_dict = cond.get("metrics", {}) or {}
        for metric in METRICS:
            for scope in ("mask", "full"):
                key = f"{metric}_{scope}"
                value = metrics_dict.get(key)
                if value is None and scope == "mask":
                    value = metrics_dict.get(metric)
                if value is None:
                    continue
                rows.append({
                    **base,
                    "eval_mask": eval_mask,
                    "intensity": intensity,
                    "metric": metric,
                    "scope": scope,
                    "value": float(value),
                })
    return rows


def load_tidy(runs_root: str, protocol: str, split: str, epoch: str, scope: str) -> pd.DataFrame:
    pattern = str(Path(runs_root).resolve() / "*" / "eval" / protocol / split / f"epoch_{epoch}" / "eval_results.json")
    rows: list[dict] = []
    for p in sorted(glob.glob(pattern)):
        rows.extend(_load_eval_file(Path(p)))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[
        (df["scope"] == scope) &
        df["eval_mask"].isin(GEOMETRY_ORDER)
    ].copy()

    df["domain"] = df["dataset"]
    df["probe"] = df["model"]
    df["geometry"] = df["eval_mask"]
    df["severity"] = df["intensity"]
    df["q_raw"] = df["value"].astype(float)
    df["q_oriented"] = df.apply(
        lambda r: r["q_raw"] if HIGHER_IS_BETTER[r["metric"]] else -r["q_raw"], axis=1
    )

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(np.nan, index=s.index)

    df["q_bar"] = df.groupby(["domain", "geometry", "metric"])["q_oriented"].transform(_minmax)
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(tidy: pd.DataFrame) -> pd.DataFrame:
    pk = ["probe", "domain", "geometry", "metric", "severity"]
    ck = ["domain", "geometry", "metric", "severity"]

    pp = tidy.groupby(pk, sort=False)[["q_bar", "q_raw"]].mean().reset_index()

    Q_m_raw = pp.groupby(ck, sort=False)["q_raw"].mean().reset_index(name="Q_m_raw")
    Q_bar   = pp.groupby(ck, sort=False)["q_bar"].mean().reset_index(name="Q_bar")
    merged  = pp.merge(Q_bar, on=ck, how="left")
    V_bar   = (
        merged.groupby(ck, sort=False)
        .apply(lambda g: float(np.mean((g["q_bar"] - g["Q_bar"]) ** 2)))
        .reset_index(name="V_bar")
    )
    n_probes = pp.groupby(ck, sort=False)["q_bar"].count().reset_index(name="n_probes")

    agg = Q_m_raw.merge(Q_bar, on=ck).merge(V_bar, on=ck).merge(n_probes, on=ck)
    agg["V_bar"] = agg["V_bar"].fillna(0.0)
    return agg


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _ordered_domains(domains) -> list[str]:
    present = set(domains)
    return [d for d in DOMAIN_ORDER if d in present] + sorted(d for d in present if d not in DOMAIN_ORDER)


def _save(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(1.035, 0.5, "M", transform=fig.transFigure, fontsize=1, alpha=0.0)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(out_path, dpi=dpi, **kw)
    pdf_dir = out_path.parent / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    fig.savefig(pdf_dir / out_path.with_suffix(".pdf").name, **kw)
    plt.close(fig)
    print(f"  saved {out_path}")


def _shared_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if not handles:
        return
    leg = fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 4), frameon=True, fancybox=False,
        edgecolor="lightgrey", facecolor="white", framealpha=1.0,
        handlelength=2.2, columnspacing=1.2, borderpad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)


def _panel_base(ax: plt.Axes, metric: str, show_xlabel: bool) -> None:
    add_train_boundary(ax)
    ax.set_title("")
    ax.set_xlabel("Mask area (%)" if show_xlabel else "")
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(axis="y", style="plain")
    ax.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)
    ax.grid(True)


def _make_grid(n_rows: int, n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    return plt.subplots(n_rows, n_cols, figsize=FIG_2x2, squeeze=False, sharex=True, sharey=False)


# ── Degradation figure ────────────────────────────────────────────────────────

def _qlabel(metric: str) -> str:
    sub = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "l1": "L1"}.get(metric, metric.upper())
    return rf"$Q_{{\mathrm{{{sub}}}}}$"


def _draw_degradation_panel(ax: plt.Axes, agg: pd.DataFrame, *, metric: str, domains: list[str], geometry: str, show_xlabel: bool) -> None:
    sub = agg[(agg["metric"] == metric) & (agg["geometry"] == geometry)].copy()
    for domain in domains:
        curve = sub[sub["domain"] == domain].groupby("severity", as_index=False)["Q_m_raw"].mean().sort_values("severity")
        if curve.empty:
            continue
        ax.plot(curve["severity"], curve["Q_m_raw"],
                color=DOMAIN_COLORS.get(domain, "#333"), linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
                marker=DOMAIN_MARKERS.get(domain, "o"), label=DOMAIN_LABELS.get(domain, domain), zorder=3)
    _panel_base(ax, metric, show_xlabel)
    ax.set_ylabel(_qlabel(metric))
    if not HIGHER_IS_BETTER.get(metric, True):
        ax.invert_yaxis()


def plot_degradation(tidy: pd.DataFrame, agg: pd.DataFrame, out_path: Path, *, dpi: int, geometry: str = "block") -> None:
    domains = _ordered_domains(tidy["domain"].unique())
    metrics = [m for m in METRICS if m in tidy["metric"].values]
    fig, axes = _make_grid(2, 2)
    for idx, metric in enumerate(metrics[:4]):
        r, c = divmod(idx, 2)
        _draw_degradation_panel(axes[r, c], agg, metric=metric, domains=domains, geometry=geometry, show_xlabel=(r == 1))
    _shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save(fig, out_path, dpi)


# ── Dispersion figure ─────────────────────────────────────────────────────────

def _slabel(metric: str) -> str:
    sub = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "l1": "L1"}.get(metric, metric.upper())
    return rf"${{\sigma}}_{{\mathrm{{{sub}}}}}$"


def _draw_dispersion_panel(ax: plt.Axes, agg: pd.DataFrame, *, metric: str, domains: list[str], geometry: str, show_xlabel: bool) -> None:
    sub = agg[(agg["metric"] == metric) & (agg["geometry"] == geometry)].copy()
    for domain in domains:
        curve = sub[sub["domain"] == domain].groupby("severity", as_index=False)["V_bar"].mean().sort_values("severity")
        if curve.empty:
            continue
        ax.plot(curve["severity"], np.sqrt(curve["V_bar"]),
                color=DOMAIN_COLORS.get(domain, "#333"), linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
                marker=DOMAIN_MARKERS.get(domain, "o"), label=DOMAIN_LABELS.get(domain, domain), zorder=3)
    _panel_base(ax, metric, show_xlabel)
    ax.set_ylabel(_slabel(metric))
    ax.set_ylim(bottom=0)


def plot_dispersion(tidy: pd.DataFrame, agg: pd.DataFrame, out_path: Path, *, dpi: int, geometry: str = "block") -> None:
    domains = _ordered_domains(tidy["domain"].unique())
    metrics = [m for m in METRICS if m in tidy["metric"].values]
    fig, axes = _make_grid(2, 2)
    for idx, metric in enumerate(metrics[:4]):
        r, c = divmod(idx, 2)
        _draw_dispersion_panel(axes[r, c], agg, metric=metric, domains=domains, geometry=geometry, show_xlabel=(r == 1))
    _shared_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save(fig, out_path, dpi)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()

    tidy = load_tidy(args.runs_root, args.protocol, args.split, args.epoch, args.scope)
    if tidy.empty:
        raise SystemExit(f"No data found under {args.runs_root!r}.")

    print(f"Loaded {tidy['source'].nunique()} runs | domains={sorted(tidy['domain'].unique())} | probes={sorted(tidy['probe'].unique())}")

    agg = aggregate(tidy)

    bad = agg[agg["n_probes"] != EXPECTED_N_PROBES]
    if not bad.empty:
        print(f"WARNING: {len(bad)} cells missing probes.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_degradation(tidy, agg, out / "degradation_all_metrics.png", dpi=args.dpi)
    plot_dispersion(tidy, agg, out / "dispersion_all_metrics.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
