"""
Plot 9 — ImageNet-Simple vs ImageNet-Complex complexity-score histogram.

Proves that the complexity-based split actually separates visually simple and
complex images.  The composite score c(I) replicates the formula in
tools/imagenet.py: z-score(variance) + z-score(sobel_mean) + z-score(edge_density),
divided by 3.

Two modes:
  1. Cache mode (fast):  read the .scores_cache.npz produced by tools/imagenet.py,
     filter paths by which subset directory they were copied into.
  2. Rescore mode (fallback): walk the val/ directories of each subset and
     score a random sample of images directly.

Usage — cache mode (preferred):
  python plots/complexity_hist.py \\
    --cache_file $DATA_PATH/imagenet/.scores_cache.npz \\
    --simple_val  $DATA_PATH/imagenet-simple/val \\
    --complex_val $DATA_PATH/imagenet-complex/val \\
    --out figures/

Usage — rescore mode (no cache required):
  python plots/complexity_hist.py \\
    --simple_val  $DATA_PATH/imagenet-simple/val \\
    --complex_val $DATA_PATH/imagenet-complex/val \\
    --out figures/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import FIG_SINGLE, savefig, set_paper_style


IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
TAU = 0.05          # edge-density threshold (matches tools/imagenet.py default)
MIN_SIZE = 100      # skip images smaller than this


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot complexity-score distributions for ImageNet-Simple vs -Complex.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--simple_val", required=True,
        help="Val directory of ImageNet-Simple ($DATA_PATH/imagenet-simple/val).",
    )
    ap.add_argument(
        "--complex_val", required=True,
        help="Val directory of ImageNet-Complex ($DATA_PATH/imagenet-complex/val).",
    )
    ap.add_argument(
        "--cache_file", default=None,
        help="Optional path to .scores_cache.npz from tools/imagenet.py.",
    )
    ap.add_argument(
        "--max_images", type=int, default=2000,
        help="Maximum images to score per subset in rescore mode.",
    )
    ap.add_argument("--out", default="figures/complexity")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ── Scoring (mirrors tools/imagenet.py) ──────────────────────────────────────

def _score_image(path: Path, tau: float = TAU) -> Optional[tuple[float, float, float]]:
    """Return (variance, sobel_mean, edge_density) or None on failure."""
    try:
        from PIL import Image
        from skimage.filters import sobel as skimage_sobel

        img = Image.open(path)
        W, H = img.size
        if min(W, H) < MIN_SIZE:
            return None
        scale = 256.0 / min(W, H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        img = img.crop(((nw - 224) // 2, (nh - 224) // 2,
                        (nw - 224) // 2 + 224, (nh - 224) // 2 + 224))
        Y = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        G = skimage_sobel(Y)
        return float(np.var(Y)), float(G.mean()), float((G > tau).mean())
    except Exception:
        return None


def _zscore(arr: np.ndarray) -> np.ndarray:
    s = arr.std()
    return (arr - arr.mean()) / s if s > 1e-10 else np.zeros_like(arr)


def _composite(v: np.ndarray, g: np.ndarray, e: np.ndarray) -> np.ndarray:
    return (_zscore(v) + _zscore(g) + _zscore(e)) / 3.0


def _collect_paths(root: Path, max_n: Optional[int], rng: random.Random) -> list[Path]:
    paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTS
    ]
    if max_n is not None and len(paths) > max_n:
        paths = rng.sample(paths, max_n)
    return paths


# ── Cache-based scoring ───────────────────────────────────────────────────────

def _scores_from_cache(
    cache_file: Path,
    simple_root: Path,
    complex_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the .npz cache and partition paths into simple vs complex by checking
    whether they are children of the respective val directories.
    """
    d = np.load(cache_file)
    paths = d["paths"]
    vs, gs, es = d["v"], d["g"], d["e"]

    simple_str  = str(simple_root.resolve())
    complex_str = str(complex_root.resolve())

    s_idx = [i for i, p in enumerate(paths) if str(Path(p).resolve()).startswith(simple_str)]
    c_idx = [i for i, p in enumerate(paths) if str(Path(p).resolve()).startswith(complex_str)]

    if not s_idx or not c_idx:
        return np.array([]), np.array([])

    # Recompute composite score over the full cache for consistent z-scoring
    scores_all = _composite(vs, gs, es)
    return scores_all[s_idx], scores_all[c_idx]


# ── Rescore mode ─────────────────────────────────────────────────────────────

def _scores_from_images(
    simple_root: Path,
    complex_root: Path,
    max_images: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    simple_paths  = _collect_paths(simple_root,  max_images, rng)
    complex_paths = _collect_paths(complex_root, max_images, rng)

    def _score_all(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vs, gs, es = [], [], []
        for p in paths:
            r = _score_image(p)
            if r is not None:
                vs.append(r[0]); gs.append(r[1]); es.append(r[2])
        return np.array(vs), np.array(gs), np.array(es)

    print(f"  scoring {len(simple_paths)} simple images ...")
    sv, sg, se = _score_all(simple_paths)
    print(f"  scoring {len(complex_paths)} complex images ...")
    cv, cg, ce = _score_all(complex_paths)

    if len(sv) == 0 or len(cv) == 0:
        return np.array([]), np.array([])

    # Z-score jointly so the split is visible on the same axis
    all_v = np.concatenate([sv, cv])
    all_g = np.concatenate([sg, cg])
    all_e = np.concatenate([se, ce])
    scores_all = _composite(all_v, all_g, all_e)
    n_s = len(sv)
    return scores_all[:n_s], scores_all[n_s:]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_hist(
    simple_scores: np.ndarray,
    complex_scores: np.ndarray,
    bins: int,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    all_scores = np.concatenate([simple_scores, complex_scores])
    bin_edges  = np.linspace(all_scores.min(), all_scores.max(), bins + 1)

    # Normalised histograms (density)
    ax.hist(
        simple_scores, bins=bin_edges,
        density=True, alpha=0.55,
        color="#0072B2", edgecolor="white", linewidth=0.4,
        label=f"ImageNet-Simple (n={len(simple_scores):,})",
        zorder=3,
    )
    ax.hist(
        complex_scores, bins=bin_edges,
        density=True, alpha=0.55,
        color="#D55E00", edgecolor="white", linewidth=0.4,
        label=f"ImageNet-Complex (n={len(complex_scores):,})",
        zorder=3,
    )

    # Optional KDE overlay (requires scipy)
    try:
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(all_scores.min(), all_scores.max(), 300)
        for scores, color in [(simple_scores, "#0072B2"), (complex_scores, "#D55E00")]:
            if len(scores) > 5:
                kde = gaussian_kde(scores, bw_method="silverman")
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5, zorder=4)
    except ImportError:
        pass

    ax.set_xlabel("Complexity score $c(I)$", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(
        "ImageNet-Simple vs ImageNet-Complex\ncomplexity score distribution",
        pad=4, fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()
    stem = "complexity_histogram"
    savefig(fig, out_dir, stem)
    print(f"  saved {out_dir / stem}.pdf/png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir     = Path(args.out)
    simple_root = Path(args.simple_val)
    complex_root = Path(args.complex_val)

    if not simple_root.exists():
        sys.exit(f"ERROR: --simple_val not found: {simple_root}")
    if not complex_root.exists():
        sys.exit(f"ERROR: --complex_val not found: {complex_root}")

    # ── Load / compute scores ────────────────────────────────────────────────
    if args.cache_file is not None:
        cache_path = Path(args.cache_file)
        if not cache_path.exists():
            print(f"  WARNING: cache file not found ({cache_path}); falling back to rescore mode.")
            simple_scores, complex_scores = _scores_from_images(
                simple_root, complex_root, args.max_images, args.seed,
            )
        else:
            print(f"  Loading scores from cache: {cache_path}")
            simple_scores, complex_scores = _scores_from_cache(
                cache_path, simple_root, complex_root,
            )
            if len(simple_scores) == 0:
                print("  Cache partition failed; falling back to rescore mode.")
                simple_scores, complex_scores = _scores_from_images(
                    simple_root, complex_root, args.max_images, args.seed,
                )
    else:
        print("  No cache file provided — rescoring images ...")
        simple_scores, complex_scores = _scores_from_images(
            simple_root, complex_root, args.max_images, args.seed,
        )

    if len(simple_scores) == 0 or len(complex_scores) == 0:
        sys.exit("ERROR: could not extract scores for one or both subsets.")

    print(f"  Simple  scores: n={len(simple_scores):<5}  mean={simple_scores.mean():.3f}  std={simple_scores.std():.3f}")
    print(f"  Complex scores: n={len(complex_scores):<5}  mean={complex_scores.mean():.3f}  std={complex_scores.std():.3f}")

    plot_hist(simple_scores, complex_scores, bins=args.bins, out_dir=out_dir)
    print(f"\nDone. Figure saved to {out_dir}/")


if __name__ == "__main__":
    main()
