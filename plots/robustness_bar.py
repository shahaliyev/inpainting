"""
Robustness summary — R_m tables (CSV) + LaTeX.

Computes the normalized AUC robustness score R_m(p, d, g) for every
(model, domain, metric) combination and writes two compact tables.

Produces (in figures/robustness/):
  robustness_table.csv      — rows: domains, columns: metrics,
                              values: model-averaged R_m
  robustness_by_model.csv   — rows: domain×metric pairs,
                              columns: U-Net, Partial Conv, Gated Conv

Also prints a LaTeX table block per metric to stdout.

Usage:
  python plots/robustness_bar.py \\
    --results (Get-ChildItem runs -Recurse -Filter eval_results.json).FullName \\
    --out figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    METRICS,
    MODEL_LABELS,
    MODEL_ORDER,
    extract_curve,
    get_xy,
    load_result,
    normalized_auc,
    result_identity,
    set_paper_style,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Robustness R_m tables (CSV + LaTeX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--family", default="block")
    return ap.parse_args()


def _compute_rm(
    runs: list[dict],
) -> dict[tuple[str, str, str], float]:
    """Return {(model, domain, metric_key): R_m}."""
    table: dict[tuple[str, str, str], float] = {}
    for r in runs:
        for candidates, _, _, higher_better in METRICS:
            xs, ys = get_xy(r["points"], candidates)
            if len(xs) < 2:
                continue
            rm = normalized_auc(xs, ys, higher_better)
            if rm is not None:
                table[(r["model"], r["domain"], candidates[0])] = rm
    return table


def _save_domain_averaged(
    rm: dict[tuple[str, str, str], float],
    domains: list[str],
    out_dir: Path,
) -> None:
    """
    robustness_table.csv:
      rows = domains, columns = metrics, values = model-averaged R_m
    """
    metric_keys  = [m[0][0] for m in METRICS]
    metric_titles = [m[1] for m in METRICS]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "robustness_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain"] + metric_titles)
        for domain in domains:
            row = [DOMAIN_LABELS.get(domain, domain)]
            for key in metric_keys:
                vals = [rm[(m, domain, key)] for m in MODEL_ORDER
                        if (m, domain, key) in rm]
                row.append(f"{float(np.mean(vals)):.4f}" if vals else "--")
            writer.writerow(row)
    print(f"  saved {path}")


def _save_by_model(
    rm: dict[tuple[str, str, str], float],
    domains: list[str],
    out_dir: Path,
) -> None:
    """
    robustness_by_model.csv:
      rows = domain×metric, columns = U-Net, Partial Conv, Gated Conv
    """
    metric_keys   = [m[0][0] for m in METRICS]
    metric_titles = {m[0][0]: m[1] for m in METRICS}
    model_cols    = [MODEL_LABELS.get(m, m) for m in MODEL_ORDER]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "robustness_by_model.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Metric"] + model_cols)
        for domain in domains:
            for key in metric_keys:
                row = [DOMAIN_LABELS.get(domain, domain), metric_titles[key]]
                for model in MODEL_ORDER:
                    v = rm.get((model, domain, key))
                    row.append(f"{v:.4f}" if v is not None else "--")
                writer.writerow(row)
    print(f"  saved {path}")


def _print_latex(
    rm: dict[tuple[str, str, str], float],
    domains: list[str],
) -> None:
    for candidates, title, _, _ in METRICS:
        key = candidates[0]
        print(f"\n% R_m — {title}")
        col_labels = " & ".join(["Model"] + [DOMAIN_LABELS.get(d, d) for d in domains])
        print(f"\\begin{{tabular}}{{l{'c' * len(domains)}}}")
        print("\\hline")
        print(col_labels + " \\\\")
        print("\\hline")
        for model in MODEL_ORDER:
            cells = [MODEL_LABELS.get(model, model)]
            for domain in domains:
                v = rm.get((model, domain, key))
                cells.append(f"{v:.3f}" if v is not None else "--")
            print(" & ".join(cells) + " \\\\")
        print("\\hline")
        print("\\end{tabular}")


def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir = Path(args.out) / "robustness"

    # ── Load ────────────────────────────────────────────────────────────────
    runs: list[dict] = []
    for p in args.results:
        data  = load_result(p)
        model, domain = result_identity(data)
        pts   = extract_curve(data, family=args.family)
        runs.append({"model": model, "domain": domain, "points": pts})
        print(f"  loaded  model={model:<14}  domain={domain}")

    domains = [d for d in DOMAIN_ORDER if any(r["domain"] == d for r in runs)] + \
              [d for d in {r["domain"] for r in runs} if d not in DOMAIN_ORDER]

    rm = _compute_rm(runs)

    _save_domain_averaged(rm, domains, out_dir)
    _save_by_model(rm, domains, out_dir)
    _print_latex(rm, domains)

    print(f"\nDone. Tables saved to {out_dir}/")


if __name__ == "__main__":
    main()
