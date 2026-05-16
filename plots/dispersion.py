"""
Cross-probe dispersion — separate plot per metric.

Generates one PNG per metric in two variants:

  Raw (unnormalized):
    Dispersion is the population std of probe values in original metric units
    at each severity level.  Values are not comparable across domains with
    different absolute metric ranges.

  Normalized:
    Raw values are min-max normalized within each (metric, domain) pair
    before computing std across probes.  This makes dispersion comparable
    across domains and metrics regardless of their absolute value range.

    Normalization formula (per metric, per domain):
      x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    where x_min/x_max span all severities × all probe models for that pair.

By default both variants are generated.

Produces (in figures/dispersion/):

  Raw (one file per metric):
    raw_dispersion_psnr.png
    raw_dispersion_ssim.png
    raw_dispersion_lpips.png
    raw_dispersion_l1.png

  Normalized (one file per metric):
    normalized_dispersion_psnr.png
    normalized_dispersion_ssim.png
    normalized_dispersion_lpips.png
    normalized_dispersion_l1.png

  normalized_dispersion_values.csv
    columns: metric, domain, severity,
             normalized_dispersion_std, normalized_dispersion_var

Flags (mutually exclusive, default: --both):
  --both          generate both raw and normalized plots and CSV (default)
  --normalized    generate only normalized plots and CSV
  --unnormalized  generate only raw plots (CSV is skipped)

Usage:
  python plots/dispersion.py --results $all --out figures
  python plots/dispersion.py --results $all --out figures --normalized
  python plots/dispersion.py --results $all --out figures --unnormalized
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    DOMAIN_LINESTYLES,
    DOMAIN_ORDER,
    FIG_2x2,
    FIG_SINGLE,
    METRICS,
    MODEL_ORDER,
    add_train_boundary,
    extract_curve,
    get_xy,
    load_result,
    result_identity,
    savefig,
    set_paper_style,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cross-probe dispersion plots (raw and normalized).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--family", default="block")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--both", action="store_true", default=False,
        help="Generate both raw and normalized plots and CSV (default behavior).",
    )
    mode.add_argument(
        "--normalized", action="store_true", default=False,
        help="Generate only normalized dispersion plots and CSV.",
    )
    mode.add_argument(
        "--unnormalized", action="store_true", default=False,
        help="Generate only raw dispersion plots. CSV is skipped.",
    )
    return ap.parse_args()


# ── Computation ───────────────────────────────────────────────────────────────

def _collect_model_values(
    by_domain_model: dict[tuple[str, str], list],
    candidates: list[str],
    domain: str,
) -> dict[str, dict[float, float]]:
    """Return {model: {severity: raw_value}} for one domain."""
    model_sev: dict[str, dict[float, float]] = {}
    for model in MODEL_ORDER:
        pts = by_domain_model.get((domain, model), [])
        xs, ys = get_xy(pts, candidates)
        if xs:
            model_sev[model] = dict(zip(xs, ys))
    return model_sev


def _std_var_at_severities(
    model_sev: dict[str, dict[float, float]],
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute population std and var across models at each common severity.
    Returns (severities, stds, vars).
    """
    if len(model_sev) < 2:
        return [], [], []
    sev_sets = [set(d.keys()) for d in model_sev.values()]
    common = sorted(sev_sets[0].intersection(*sev_sets[1:]))
    if not common:
        return [], [], []
    stds, vars_ = [], []
    for s in common:
        vals = np.array([model_sev[m][s] for m in model_sev if s in model_sev[m]])
        stds.append(float(np.std(vals, ddof=0)))
        vars_.append(float(np.var(vals, ddof=0)))
    return common, stds, vars_


def compute_raw(
    by_domain_model: dict[tuple[str, str], list],
    candidates: list[str],
    domains: list[str],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """Raw (unnormalized) cross-probe dispersion. Returns {domain: (sevs, stds, vars)}."""
    result = {}
    for domain in domains:
        model_sev = _collect_model_values(by_domain_model, candidates, domain)
        sevs, stds, vars_ = _std_var_at_severities(model_sev)
        if sevs:
            result[domain] = (sevs, stds, vars_)
    return result


def compute_normalized(
    by_domain_model: dict[tuple[str, str], list],
    candidates: list[str],
    domains: list[str],
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Normalized cross-probe dispersion.
    Min-max normalization within (metric, domain) before computing std.
    Returns {domain: (sevs, stds, vars)}.
    """
    result = {}
    for domain in domains:
        model_sev_raw = _collect_model_values(by_domain_model, candidates, domain)
        if len(model_sev_raw) < 2:
            continue
        all_raw = [v for d in model_sev_raw.values() for v in d.values()]
        x_min  = float(min(all_raw))
        x_max  = float(max(all_raw))
        denom  = x_max - x_min + 1e-8
        model_sev_norm = {
            model: {s: (v - x_min) / denom for s, v in sev_map.items()}
            for model, sev_map in model_sev_raw.items()
        }
        sevs, stds, vars_ = _std_var_at_severities(model_sev_norm)
        if sevs:
            result[domain] = (sevs, stds, vars_)
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(
    ax: plt.Axes,
    title: str,
    y_label: str,
    domain_data: dict[str, tuple[list, list, list]],
    domains: list[str],
    family: str,
) -> tuple[list, list[str]]:
    handles, labels = [], []
    for domain in domains:
        if domain not in domain_data:
            continue
        sevs, stds, _ = domain_data[domain]
        if not sevs:
            continue
        line, = ax.plot(
            sevs, stds,
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            linewidth=1.4, marker="o", markersize=3,
            zorder=3,
        )
        handles.append(line)
        labels.append(DOMAIN_LABELS.get(domain, domain))

    add_train_boundary(ax)
    ax.set_xlabel("Mask area (%)" if family != "freeform" else "Strokes", fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.set_title(title, pad=3)
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.set_ylim(bottom=0)
    ax.grid(True)
    return handles, labels


def _save_panel(
    title: str,
    y_label: str,
    domain_data: dict[str, tuple[list, list, list]],
    domains: list[str],
    family: str,
    stem: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)
    h, l = _draw_panel(ax, title, y_label, domain_data, domains, family)
    if h:
        ax.legend(h, l, fontsize=7, frameon=True)
    plt.tight_layout()
    savefig(fig, out_dir, stem)


def _save_2x2(
    by_metric: dict[str, dict[str, tuple[list, list, list]]],
    y_label: str,
    title_prefix: str,
    stem: str,
    domains: list[str],
    family: str,
    out_dir: Path,
) -> None:
    """Save a 2×2 combined figure — one panel per metric, one global legend."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_2x2)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    global_handles, global_labels = [], []

    for ax, (candidates, title, _, _) in zip(axes_flat, METRICS):
        key = candidates[0]
        domain_data = by_metric.get(key, {})
        h, l = _draw_panel(ax, f"{title_prefix}{title}", y_label, domain_data, domains, family)
        for handle, label in zip(h, l):
            if label not in global_labels:
                global_handles.append(handle)
                global_labels.append(label)

    fig.legend(
        global_handles, global_labels,
        loc="lower center", ncol=len(global_handles),
        bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, out_dir, stem)


# ── CSV ───────────────────────────────────────────────────────────────────────

def _save_csv(
    norm_by_metric: dict[str, dict[str, tuple[list, list, list]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "normalized_dispersion_values.csv"
    fieldnames = [
        "metric", "domain", "severity",
        "normalized_dispersion_std", "normalized_dispersion_var",
    ]
    rows = []
    for metric_key, domain_data in norm_by_metric.items():
        for domain, (sevs, stds, vars_) in domain_data.items():
            for s, std, var in zip(sevs, stds, vars_):
                rows.append({
                    "metric":                    metric_key,
                    "domain":                    domain,
                    "severity":                  f"{s:.1f}",
                    "normalized_dispersion_std": f"{std:.6f}",
                    "normalized_dispersion_var": f"{var:.6f}",
                })
    rows.sort(key=lambda r: (r["metric"], r["domain"], float(r["severity"])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir = Path(args.out) / "dispersion"

    do_normalized   = not args.unnormalized
    do_unnormalized = not args.normalized

    # ── Load ────────────────────────────────────────────────────────────────
    by_domain_model: dict[tuple[str, str], list] = {}
    for p in args.results:
        data  = load_result(p)
        model, domain = result_identity(data)
        pts   = extract_curve(data, family=args.family)
        by_domain_model[(domain, model)] = pts
        print(f"  loaded  model={model:<14}  domain={domain}")

    present = {k[0] for k in by_domain_model}
    domains = [d for d in DOMAIN_ORDER if d in present] + \
              sorted(present - set(DOMAIN_ORDER))

    # ── Generate plots ───────────────────────────────────────────────────────
    norm_by_metric: dict[str, dict] = {}
    raw_by_metric:  dict[str, dict] = {}

    for candidates, title, _, _ in METRICS:
        key = candidates[0]

        if do_unnormalized:
            raw_data = compute_raw(by_domain_model, candidates, domains)
            raw_by_metric[key] = raw_data
            _save_panel(
                title=f"Cross-probe dispersion: {title}",
                y_label="$\\sigma$ (cross-probe)",
                domain_data=raw_data,
                domains=domains,
                family=args.family,
                stem=f"raw_dispersion_{key}",
                out_dir=out_dir,
            )

        if do_normalized:
            norm_data = compute_normalized(by_domain_model, candidates, domains)
            norm_by_metric[key] = norm_data
            _save_panel(
                title=f"Normalized dispersion: {title}",
                y_label="Normalized $\\sigma$ (cross-probe)",
                domain_data=norm_data,
                domains=domains,
                family=args.family,
                stem=f"normalized_dispersion_{key}",
                out_dir=out_dir,
            )

    # ── Combined 2×2 figures ─────────────────────────────────────────────────
    if do_unnormalized and raw_by_metric:
        _save_2x2(
            by_metric=raw_by_metric,
            y_label="$\\sigma$ (cross-probe)",
            title_prefix="",
            stem="raw_dispersion_all",
            domains=domains,
            family=args.family,
            out_dir=out_dir,
        )

    if do_normalized and norm_by_metric:
        _save_2x2(
            by_metric=norm_by_metric,
            y_label="Normalized $\\sigma$",
            title_prefix="Norm. ",
            stem="normalized_dispersion_all",
            domains=domains,
            family=args.family,
            out_dir=out_dir,
        )
        _save_csv(norm_by_metric, out_dir)

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
