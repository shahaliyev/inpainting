"""
Paired t-test analysis of placement effects on inpainting quality.

For each (domain, metric, severity) cell, runs a paired two-sample t-test
across every C(5,2)=10 placement pairs.  The same images appear at every
placement, so pairing on (image, model) removes image-level variance and
gives maximum statistical power.

Output
------
  placement_ttest.csv   one row per (domain, metric, severity, pair)

Columns
-------
  metric, domain, mask_severity,
  placement_a, placement_b,
  n_subjects,           number of (image, model) pairs used
  mean_a, mean_b,       group means
  mean_diff,            mean_a - mean_b
  std_diff,             std of per-pair differences
  t_stat,               paired t-statistic
  df,                   degrees of freedom (n - 1)
  p_raw,                two-sided p-value before correction
  p_bonferroni,         Bonferroni-adjusted p-value (10 comparisons per cell)
  cohens_d,             mean_diff / std_diff (paired Cohen's d)
  significant_05_adj,   p_bonferroni < 0.05
  significant_01_adj,   p_bonferroni < 0.01

Usage
-----
  python placement/ttest.py
  python placement/ttest.py --severity 10 30
  python placement/ttest.py --domain dtd carpet --metric psnr ssim
  python placement/ttest.py --data_dir runs/placement --out_dir figures/placement
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Constants ─────────────────────────────────────────────────────────────────

METRICS = [
    ("psnr",  True),
    ("ssim",  True),
    ("lpips", False),
    ("l1",    False),
]
METRIC_NAMES = [m[0] for m in METRICS]
HIGHER_IS_BETTER = {m[0]: m[1] for m in METRICS}

PLACEMENT_ORDER = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
DOMAIN_ORDER    = ["carpet", "dtd", "imagenet-simple", "imagenet-complex"]

N_PAIRS = 10   # C(5, 2)

_GENERATED_STEMS = {
    "placement_summary", "placement_stats", "placement_posthoc",
    "placement_lmm_global", "placement_anova_global", "placement_ttest",
}
_REQUIRED_COLS = {
    "model", "domain", "image", "mask_severity", "mask_placement",
    "l1", "psnr", "ssim", "lpips",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Paired t-test analysis of placement effects on inpainting metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data_dir", default=str(REPO_ROOT / "runs" / "placement"),
        help="Directory containing per-image placement *.csv files.",
    )
    ap.add_argument(
        "--out_dir", default=str(REPO_ROOT / "figures" / "placement"),
        help="Output directory for placement_ttest.csv.",
    )
    ap.add_argument(
        "--severity", type=int, nargs="+", default=None,
        help="Restrict to these mask severities (e.g. --severity 10 30).",
    )
    ap.add_argument(
        "--domain", nargs="+", default=None,
        help="Restrict to these domains.",
    )
    ap.add_argument(
        "--model", nargs="+", default=None,
        help="Restrict to these model names.",
    )
    ap.add_argument(
        "--metric", nargs="+", default=None,
        choices=METRIC_NAMES,
        help="Restrict to these metrics. Default: all.",
    )
    ap.add_argument(
        "--min_subjects", type=int, default=10,
        help="Minimum paired subjects required to run a test.",
    )
    return ap.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate raw per-image placement CSVs, skipping generated files."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    skipped: list[str] = []

    for f in csv_files:
        if f.stem in _GENERATED_STEMS:
            skipped.append(f"{f.name} (generated — skipped)")
            continue

        try:
            header = pd.read_csv(f, nrows=0)
        except Exception as e:
            skipped.append(f"{f.name} (read error: {e})")
            continue

        missing = _REQUIRED_COLS - set(header.columns)
        if missing:
            skipped.append(f"{f.name} (missing: {sorted(missing)})")
            continue

        df = pd.read_csv(f)
        for col in ("l1", "psnr", "ssim", "lpips"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs.append(df)

    if skipped:
        print(f"  Skipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"    * {s}")

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


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_ttest(df: pd.DataFrame, metrics: list[str], min_subjects: int) -> pd.DataFrame:
    """
    Paired t-test for every (domain, metric, severity, placement_pair).

    Pairs subjects by (image, model) — the same image processed by the same
    model at two different placements.  Subjects missing at either placement
    are dropped (complete-case pairing).
    """
    rows: list[dict] = []

    for metric in metrics:
        for domain in _ordered_domains(df["domain"].unique()):
            dom_df = df[df["domain"] == domain]

            for severity in sorted(dom_df["mask_severity"].unique()):
                sev_df = dom_df[dom_df["mask_severity"] == severity]

                placements_present = [
                    p for p in PLACEMENT_ORDER
                    if p in sev_df["mask_placement"].cat.categories
                ]
                if len(placements_present) < 2:
                    continue

                # Block by (image, model) to pair each image-model observation.
                block_cols = [c for c in ("image", "model") if c in sev_df.columns]
                if not block_cols:
                    continue

                pivot = (
                    sev_df.pivot_table(
                        index=block_cols,
                        columns="mask_placement",
                        values=metric,
                        aggfunc="mean",
                    )
                    .reindex(columns=placements_present)
                )

                for pl_a, pl_b in combinations(placements_present, 2):
                    paired = pivot[[pl_a, pl_b]].dropna()
                    n = len(paired)

                    if n < min_subjects:
                        continue

                    x = paired[pl_a].values
                    y = paired[pl_b].values
                    diff = x - y

                    mean_a   = float(np.mean(x))
                    mean_b   = float(np.mean(y))
                    mean_diff = float(np.mean(diff))
                    std_diff  = float(np.std(diff, ddof=1))

                    # Paired t-test
                    try:
                        t_stat, p_raw = ttest_rel(x, y)
                        t_stat = float(t_stat)
                        p_raw  = float(p_raw)
                    except Exception:
                        t_stat = p_raw = float("nan")

                    # Bonferroni correction (N_PAIRS comparisons per cell)
                    p_adj = (
                        float(np.clip(p_raw * N_PAIRS, 0.0, 1.0))
                        if not np.isnan(p_raw) else float("nan")
                    )

                    # Cohen's d for paired design = mean(diff) / std(diff)
                    cohens_d = (
                        float(mean_diff / std_diff)
                        if std_diff > 1e-10 else float("nan")
                    )

                    rows.append({
                        "metric":           metric,
                        "higher_is_better": HIGHER_IS_BETTER[metric],
                        "domain":           domain,
                        "mask_severity":    int(severity),
                        "placement_a":      pl_a,
                        "placement_b":      pl_b,
                        "n_subjects":       n,
                        "mean_a":           round(mean_a,   6),
                        "mean_b":           round(mean_b,   6),
                        "mean_diff":        round(mean_diff, 6),
                        "std_diff":         round(std_diff,  6),
                        "t_stat":           round(t_stat,    4) if not np.isnan(t_stat) else float("nan"),
                        "df":               n - 1,
                        "p_raw":            p_raw,
                        "p_bonferroni":     p_adj,
                        "cohens_d":         round(cohens_d, 4) if not np.isnan(cohens_d) else float("nan"),
                        "significant_05_adj": bool(p_adj < 0.05) if not np.isnan(p_adj) else False,
                        "significant_01_adj": bool(p_adj < 0.01) if not np.isnan(p_adj) else False,
                    })

    return pd.DataFrame(rows)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No tests produced.")
        return

    total = len(df)
    sig05 = int(df["significant_05_adj"].sum())
    sig01 = int(df["significant_01_adj"].sum())

    print(f"\n  Total tests   : {total}")
    print(f"  Significant (Bonferroni-adjusted):")
    print(f"    alpha=0.05  : {sig05}/{total} ({100*sig05/total:.1f}%)")
    print(f"    alpha=0.01  : {sig01}/{total} ({100*sig01/total:.1f}%)")

    # Largest effect sizes
    if "cohens_d" in df.columns:
        top = (
            df.dropna(subset=["cohens_d"])
            .assign(abs_d=lambda x: x["cohens_d"].abs())
            .nlargest(5, "abs_d")[
                ["metric", "domain", "mask_severity",
                 "placement_a", "placement_b", "cohens_d", "p_bonferroni"]
            ]
        )
        if not top.empty:
            print("\n  Top-5 largest |Cohen's d|:")
            print(top.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    df = load_data(args.data_dir)
    if df.empty:
        raise SystemExit(
            f"No placement CSV files found under: {args.data_dir}\n"
            "Run eval_placement.py first."
        )

    if args.domain:
        df = df[df["domain"].isin(args.domain)]
    if args.model:
        df = df[df["model"].isin(args.model)]
    if args.severity:
        df = df[df["mask_severity"].isin(args.severity)]
    if df.empty:
        raise SystemExit("No data remains after applying filters.")

    metrics = args.metric or [m for m in METRIC_NAMES if m in df.columns]

    domains    = _ordered_domains(df["domain"].unique())
    severities = sorted(df["mask_severity"].unique())
    models     = sorted(df["model"].unique())

    print(
        f"Loaded {len(df):,} rows from {args.data_dir}"
        f"\n  domains   : {domains}"
        f"\n  models    : {models}"
        f"\n  severities: {severities}"
        f"\n  metrics   : {metrics}"
        f"\n  output    : {args.out_dir}"
    )

    print("\nRunning paired t-tests...")
    result = run_ttest(df, metrics, args.min_subjects)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not result.empty:
        path = out / "placement_ttest.csv"
        result.to_csv(path, index=False)
        print(f"\n  Saved -> {path}  ({len(result)} rows)")
    else:
        print("  No results produced (check --min_subjects or data availability).")

    _print_summary(result)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
