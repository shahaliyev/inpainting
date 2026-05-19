"""
Shared utilities for all plots/ scripts.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"

# ── Metrics ─────────────────────────────────────────────────────────────────
# Each entry: (key_candidates, short_title, y_axis_label, higher_is_better)
METRICS: list[tuple[list[str], str, str, bool]] = [
    (["psnr", "psnr_mask", "psnr_full"],   "PSNR",      "PSNR (dB)",   True),
    (["ssim", "ssim_mask", "ssim_full"],   "SSIM",      "SSIM",        True),
    (["lpips", "lpips_mask", "lpips_full"],"LPIPS",     "LPIPS",       False),
    (["l1", "l1_mask", "l1_full"],         "L1",        "L1",          False),
]
METRIC_CANONICAL = [m[0][0] for m in METRICS]   # ["psnr", "ssim", "lpips", "l1"]
HIGHER_IS_BETTER: dict[str, bool] = {
    candidates[0]: higher_better
    for candidates, _, _, higher_better in METRICS
}

# ── Domain ordering and display labels ──────────────────────────────────────
DOMAIN_ORDER  = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]
DOMAIN_LABELS: dict[str, str] = {
    "carpet":           "Carpet",
    "dtd":              "DTD",
    "imagenet-simple":  "ImageNet-Simple",
    "imagenet-complex": "ImageNet-Complex",
}

# ── Model ordering and display labels ───────────────────────────────────────
MODEL_ORDER  = ["unet", "partial_conv", "gated_conv"]
MODEL_LABELS: dict[str, str] = {
    "unet":         "U-Net",
    "partial_conv": "Partial Conv",
    "gated_conv":   "Gated Conv",
}

# ── Colorblind-safe palette (Okabe–Ito) ─────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "unet":         "#0072B2",   # blue
    "partial_conv": "#D55E00",   # vermilion
    "gated_conv":   "#009E73",   # bluish-green
}
MODEL_MARKERS: dict[str, str] = {
    "unet":         "o",
    "partial_conv": "s",
    "gated_conv":   "^",
}

# Domain colors/linestyles for dispersion plots (same Okabe–Ito extension)
DOMAIN_COLORS: dict[str, str] = {
    "carpet":           "#0072B2",
    "dtd":              "#D55E00",
    "imagenet-simple":  "#009E73",
    "imagenet-complex": "#CC79A7",
}
DOMAIN_LINESTYLES: dict[str, str] = {
    "carpet":           "-",
    "dtd":              "--",
    "imagenet-simple":  "-.",
    "imagenet-complex": ":",
}

# Maximum training severity — shown as a subtle dashed vertical line
TRAIN_SEVERITY_MAX = 30

# ── Figure sizes (inches, IEEE double-column text width ≈ 7.16 in) ──────────
FIG_SINGLE  = (3.5,  2.6)   # one journal column
FIG_DOUBLE  = (7.0,  2.6)   # two journal columns
FIG_2x2     = (7.0,  5.4)   # 2×2 panel grid


# ── Paper style ───────────────────────────────────────────────────────────────

def set_paper_style() -> None:
    """Apply publication-quality rcParams. Call once at the start of each script."""
    mpl.rcParams.update({
        "font.family":          "serif",
        "mathtext.fontset":     "cm",
        "font.size":            8,
        "axes.labelsize":       12,
        "axes.titlesize":       8,
        "legend.fontsize":      7,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "axes.linewidth":       0.7,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.3,
        "lines.linewidth":      1.4,
        "lines.markersize":     3.5,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_result(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def result_identity(result: dict) -> tuple[str, str]:
    """Return (model_name, dataset_name)."""
    return result.get("model", "unknown"), result.get("dataset", "unknown")


def discover_eval_paths(
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
) -> list[Path]:
    """Find eval_results.json files under the standard runs/ layout."""
    runs_root = Path(runs_root).resolve()
    pattern = str(
        runs_root / "*" / "eval" / protocol / split / f"epoch_{epoch}" / "eval_results.json"
    )
    return [Path(p) for p in sorted(glob.glob(pattern))]


def resolve_eval_paths(
    results: list[str | Path] | None = None,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
) -> list[Path]:
    """Use explicit eval paths when provided, otherwise discover them from runs/."""
    if results:
        return [Path(p).resolve() for p in results]
    return discover_eval_paths(
        runs_root=runs_root,
        protocol=protocol,
        split=split,
        epoch=epoch,
    )


def load_by_domain_model(
    results: list[str | Path] | None = None,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
    family: str = "block",
) -> tuple[dict[tuple[str, str], list[tuple[float, dict]]], list[str]]:
    """Load eval curves as {(domain, model): points}, plus ordered domains."""
    by_domain_model: dict[tuple[str, str], list[tuple[float, dict]]] = {}
    for path in resolve_eval_paths(
        results,
        runs_root=runs_root,
        protocol=protocol,
        split=split,
        epoch=epoch,
    ):
        data = load_result(path)
        model, domain = result_identity(data)
        by_domain_model[(domain, model)] = extract_curve(data, family=family)
        print(f"  loaded  model={model:<14}  domain={domain}")

    present = {domain for domain, _ in by_domain_model}
    domains = ordered_domains(present)
    return by_domain_model, domains


# ── Curve extraction ──────────────────────────────────────────────────────────

def extract_curve(
    result: dict,
    family: str = "block",
) -> list[tuple[float, dict]]:
    """
    Return [(severity_pct, metrics_dict), ...] sorted by severity.
    For freeform, severity = num_strokes.
    """
    points: list[tuple[float, dict]] = []
    for cond in result.get("conditions", []):
        mask_type = Path(cond.get("mask_yaml", "")).stem
        metrics   = cond.get("metrics") or {}
        if mask_type == family:
            if family == "freeform":
                n = (cond.get("mask_overrides") or {}).get("num_strokes")
                if n is not None:
                    points.append((float(n), metrics))
            else:
                ratios = cond.get("mask_ratios") or []
                if ratios:
                    points.append((float(ratios[0]), metrics))
    points.sort(key=lambda x: x[0])
    return points


def get_xy(
    points: list[tuple[float, dict]],
    candidates: list[str],
) -> tuple[list[float], list[float]]:
    """Extract (xs, ys) using the first available metric key."""
    key = next((c for c in candidates if any(c in m for _, m in points)), None)
    if key is None:
        return [], []
    xs, ys = [], []
    for x, m in points:
        y = m.get(key)
        if y is not None:
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def _parse_condition(cond: dict) -> dict:
    """Pull eval_mask + intensity out of one evaluation condition."""
    mask_yaml = cond.get("mask_yaml", "") or ""
    eval_mask = Path(mask_yaml).stem or "unknown"
    mask_ratios = cond.get("mask_ratios") or []

    if eval_mask in ("block", "multi_block") and mask_ratios:
        return {
            "eval_mask": eval_mask,
            "intensity": float(mask_ratios[0]),
            "intensity_kind": "ratio",
        }

    if eval_mask == "freeform":
        n_strokes = (cond.get("mask_overrides") or {}).get("num_strokes")
        if n_strokes is not None:
            return {
                "eval_mask": eval_mask,
                "intensity": float(n_strokes),
                "intensity_kind": "strokes",
            }

    return {
        "eval_mask": eval_mask,
        "intensity": float("nan"),
        "intensity_kind": "unknown",
    }


def load_eval_file_long(path: str | Path) -> list[dict]:
    """Read one eval_results.json into long-format metric rows."""
    path = Path(path).resolve()
    data = load_result(path)

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

        for metric in METRIC_CANONICAL:
            for scope in ("mask", "full"):
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
    results: list[str | Path] | None = None,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
) -> pd.DataFrame:
    """Load all requested/discovered eval JSONs into a long DataFrame."""
    rows: list[dict] = []
    for path in resolve_eval_paths(
        results,
        runs_root=runs_root,
        protocol=protocol,
        split=split,
        epoch=epoch,
    ):
        rows.extend(load_eval_file_long(path))
    return pd.DataFrame(rows)


def _normalize_series(values: pd.Series) -> pd.Series:
    """Min-max normalize one oriented curve."""
    y_min = float(values.min())
    y_max = float(values.max())
    if y_max > y_min:
        return (values - y_min) / (y_max - y_min)
    return pd.Series(np.nan, index=values.index, dtype=float)


def orient_value(metric: str, value: float) -> float:
    """Orient one metric value so that larger is better."""
    if HIGHER_IS_BETTER[metric]:
        return float(value)
    return float(-value)


def load_tidy_evals(
    results: list[str | Path] | None = None,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    protocol: str = "degradation_v1",
    split: str = "val",
    epoch: str = "*",
    scope: str = "mask",
    geometries: tuple[str, ...] = ("block",),
    severity_kind: str = "ratio",
) -> pd.DataFrame:
    """Return tidy eval rows with q_raw, q_oriented, and q_bar columns."""
    df = load_all_evals(
        results,
        runs_root=runs_root,
        protocol=protocol,
        split=split,
        epoch=epoch,
    )

    if df.empty:
        return df

    df = df[df["scope"] == scope].copy()
    df = df[
        df["eval_mask"].isin(geometries)
        & (df["intensity_kind"] == severity_kind)
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
    )["q_oriented"].transform(_normalize_series)

    return df


# ── Math ──────────────────────────────────────────────────────────────────────

def orient(ys: list[float], higher_better: bool) -> list[float]:
    """Orient so that higher always means better reconstruction."""
    return list(ys) if higher_better else [-y for y in ys]


def normalize(ys: list[float]) -> list[float]:
    """Min-max normalize a single oriented degradation curve to q_bar."""
    if not ys:
        return []
    y_min, y_max = min(ys), max(ys)
    if abs(y_max - y_min) < 1e-12:
        return [float("nan") for _ in ys]
    return [(y - y_min) / (y_max - y_min) for y in ys]


def oriented_normalized_curve(
    xs: list[float],
    ys: list[float],
    higher_better: bool,
) -> tuple[list[float], list[float]]:
    """Return (severity, q_bar) after orientation and curve-wise normalization."""
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    if not pairs:
        return [], []
    xs_sorted = [float(x) for x, _ in pairs]
    ys_sorted = [float(y) for _, y in pairs]
    return xs_sorted, normalize(orient(ys_sorted, higher_better))


def ordered_domains(domains) -> list[str]:
    """Order domains by the paper's canonical order, then alphabetically."""
    present = list(domains)
    ordered = [d for d in DOMAIN_ORDER if d in present]
    ordered += sorted(d for d in present if d not in ordered)
    return ordered


def ordered_models(models) -> list[str]:
    """Order models by the paper's canonical order, then alphabetically."""
    present = list(models)
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += sorted(m for m in present if m not in ordered)
    return ordered


def aggregate_q_v(tidy: pd.DataFrame) -> pd.DataFrame:
    """Compute Q_bar and V_bar from tidy q_bar rows."""
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
        .apply(lambda g: np.nanmean((g["q_bar"] - g["Q_bar"]) ** 2))
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


def build_curves_long(tidy: pd.DataFrame, q_agg: pd.DataFrame) -> pd.DataFrame:
    """Export q_bar, Q_bar, and V_bar in long form."""
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

    long = per_probe.merge(
        q_agg[cell_keys + ["Q_bar", "V_bar", "n_probes"]],
        on=cell_keys,
        how="left",
    )

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


def build_dispersion_table(q_agg: pd.DataFrame) -> pd.DataFrame:
    """Create the per-severity V_bar table used for reporting."""
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
    """Summarize V_bar across severity levels."""
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


def compute_r_m_from_qbar(severities: np.ndarray, q_bar: np.ndarray) -> float:
    """Compute normalized AUC R_m from an already-normalized q_bar curve."""
    order = np.argsort(severities)
    x = severities[order].astype(float)
    y = q_bar[order].astype(float)
    if len(x) < 2 or np.isnan(y).any():
        return float("nan")
    span = float(x[-1] - x[0])
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(y, x) / span)


def build_robustness_table(tidy: pd.DataFrame) -> pd.DataFrame:
    """Compute R_m for each probe, domain, geometry, and metric."""
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
        rec = dict(zip(group_cols, key))
        rec["R_m"] = compute_r_m_from_qbar(
            g["severity"].to_numpy(dtype=float),
            g["q_bar"].to_numpy(dtype=float),
        )
        rec["n_points"] = int(len(g))
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

    return pd.concat([per_probe, pd.DataFrame(mean_rows)], ignore_index=True)


# ── Save helper ───────────────────────────────────────────────────────────────

def savefig(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    """Save PNG and PDF. PDFs go into a pdf/ subfolder of out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png)
    print(f"  saved {png}")
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / f"{stem}.pdf"
    fig.savefig(pdf)
    print(f"  saved {pdf}")
    plt.close(fig)


def add_train_boundary(ax: mpl.axes.Axes) -> None:
    """Draw the subtle dashed vertical line at the training severity maximum."""
    ax.axvline(
        x=TRAIN_SEVERITY_MAX,
        color="#aaaaaa",
        linewidth=0.7,
        linestyle="--",
        zorder=2,
    )
