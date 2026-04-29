import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.build import build_dataloader
from evaluation.config import load_dataset_and_mask, merge_mask_cfg, resolve_config_path
from evaluation.grid import get_eval_grid
from models.build import build_model
from training.checkpoint import validate_checkpoint_schema
from training.engine import evaluate
from training.logger import MetricsLogger
from utils.config_resolver import require_cfg_fields
from utils.runtime_messages import cfg_name, startup_summary_line

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", default=None, help="Eval config key under configs/eval, e.g. default")
    ap.add_argument("--eval_yaml", default=None, help="Optional explicit eval YAML path (advanced)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--no_lpips", action="store_true", help="Skip LPIPS computation (saves time on CPU)")
    ap.add_argument("--metric_scope", choices=["mask", "full"], default=None,
                    help="Override metric scope. Defaults to train_cfg.metrics.scope or 'mask'.")
    ap.add_argument("--report_both_metrics", action="store_true",
                    help="Also report both mask/full metric variants in eval_results.json.")
    ap.add_argument("--strict_config_match", action="store_true",
                    help="Fail if eval config paths mismatch checkpoint metadata defaults.")
    return ap.parse_args()


def _cfg_from_ckpt_raw(ckpt_raw, key):
    if key not in ckpt_raw or ckpt_raw[key] is None:
        raise ValueError(f"Checkpoint is missing required '{key}'. Re-train with latest train.py.")
    return OmegaConf.create(ckpt_raw[key])


def apply_cli_overrides(args, dataset_cfg, loader_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = False
        dataset_cfg.limit_seed = int(args.seed)
    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)


def infer_eval_dir_from_ckpt(ckpt_path: Path, eval_profile: str, split: str, epoch: int) -> Path:
    if ckpt_path.parent.name != "checkpoints":
        raise ValueError(f"Checkpoint path must be inside a 'checkpoints' directory: {ckpt_path}")
    train_run_dir = ckpt_path.parent.parent
    return train_run_dir / "eval" / eval_profile / split / f"epoch_{int(epoch)}"


def build_default_grid_from_training_mask(dataset_cfg_base, mask_cfg_base, dataset_path, mask_path):
    mask_name = str(getattr(mask_cfg_base, "name", "unknown")).lower()
    if mask_name != "mixed":
        return [{
            "name": mask_name if mask_name else "default",
            "dataset_cfg": dataset_cfg_base,
            "mask_cfg": mask_cfg_base,
            "mask_ratios": None,
            "mask_overrides": None,
            "dataset_yaml": dataset_path,
            "mask_yaml": mask_path,
        }]

    if not dataset_path:
        # Fallback for older checkpoints lacking config_paths metadata.
        return [{
            "name": "mixed",
            "dataset_cfg": dataset_cfg_base,
            "mask_cfg": mask_cfg_base,
            "mask_ratios": None,
            "mask_overrides": None,
            "dataset_yaml": None,
            "mask_yaml": None,
        }]

    # For mixed-trained models, evaluate each constituent family separately by default.
    return [
        {
            "name": "block",
            "dataset_yaml": dataset_path,
            "mask_yaml": resolve_config_path("mask", "block"),
            "mask_ratios": None,
            "mask_overrides": None,
        },
        {
            "name": "multi_block",
            "dataset_yaml": dataset_path,
            "mask_yaml": resolve_config_path("mask", "multi_block"),
            "mask_ratios": None,
            "mask_overrides": None,
        },
        {
            "name": "freeform",
            "dataset_yaml": dataset_path,
            "mask_yaml": resolve_config_path("mask", "freeform"),
            "mask_ratios": None,
            "mask_overrides": None,
        },
    ]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.eval and args.eval_yaml:
        raise ValueError("Use either --eval or --eval_yaml, not both.")
    eval_yaml = args.eval_yaml or (resolve_config_path("eval", args.eval) if args.eval else None)

    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    validate_checkpoint_schema(ckpt_raw)
    model_cfg = _cfg_from_ckpt_raw(ckpt_raw, "model_cfg")
    model = build_model(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(ckpt_raw["model"])

    loss_fn = nn.L1Loss(reduction="none")
    dataset_cfg_base = _cfg_from_ckpt_raw(ckpt_raw, "dataset_cfg")
    loader_cfg_base = _cfg_from_ckpt_raw(ckpt_raw, "loader_cfg")
    mask_cfg_base = _cfg_from_ckpt_raw(ckpt_raw, "mask_cfg")
    train_cfg = _cfg_from_ckpt_raw(ckpt_raw, "train_cfg")
    require_cfg_fields(dataset_cfg_base, ["norm.mean", "norm.std"], "checkpoint dataset config")
    require_cfg_fields(loader_cfg_base, ["batch_size"], "checkpoint loader config")
    require_cfg_fields(train_cfg, ["metrics.scope", "metrics.report_both"], "checkpoint train config")
    state_epoch = int(ckpt_raw.get("epoch", 0))
    state_step = int(ckpt_raw.get("step", 0))

    if "${oc.env:" in str(getattr(dataset_cfg_base, "root", "")):
        OmegaConf.resolve(dataset_cfg_base)

    use_amp = bool(getattr(train_cfg, "mixed_precision", False)) and device.type == "cuda"
    train_metrics_cfg = getattr(train_cfg, "metrics", OmegaConf.create({}))
    metric_scope = args.metric_scope or str(getattr(train_metrics_cfg, "scope", "mask")).lower()
    if metric_scope not in {"mask", "full"}:
        raise ValueError(f"Unsupported metric_scope: {metric_scope}. Use 'mask' or 'full'.")
    report_both_metrics = bool(getattr(train_metrics_cfg, "report_both", True)) or bool(args.report_both_metrics)
    eval_profile = Path(eval_yaml).stem if eval_yaml else "default"
    run_dir = infer_eval_dir_from_ckpt(ckpt_path, eval_profile, args.split, state_epoch)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir) if args.save_vis else None

    if args.no_lpips:
        lpips_net = None
    else:
        if device.type == "cpu":
            print("WARNING: LPIPS is enabled on CPU — this will be slow. Pass --no_lpips to skip it.")
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(device).eval()

    config_paths = ckpt_raw.get("config_paths", {}) or {}
    dataset_path = config_paths.get("dataset_yaml")
    mask_path = config_paths.get("mask_yaml")
    model_path = config_paths.get("model_yaml")
    if args.strict_config_match:
        missing = [k for k, v in {"dataset_yaml": dataset_path, "mask_yaml": mask_path, "model_yaml": model_path}.items() if not v]
        if missing:
            raise ValueError(f"Strict mode requires checkpoint config_paths keys, missing: {missing}")

    if eval_yaml:
        eval_cfg = OmegaConf.load(eval_yaml)
        default_dataset_yaml = dataset_path
        default_mask_yaml = mask_path
        if not default_dataset_yaml or not default_mask_yaml:
            raise ValueError("Checkpoint metadata missing config_paths.dataset_yaml/mask_yaml for eval grid usage.")
        grid = get_eval_grid(eval_cfg, default_dataset_yaml, default_mask_yaml)
        if args.strict_config_match:
            for cond in grid:
                if cond.get("dataset_yaml") and Path(cond["dataset_yaml"]).stem != Path(default_dataset_yaml).stem:
                    raise ValueError(
                        f"Strict mode mismatch: eval condition dataset '{cond['dataset_yaml']}' "
                        f"differs from checkpoint dataset '{default_dataset_yaml}'."
                    )
                if cond.get("mask_yaml") and Path(cond["mask_yaml"]).stem != Path(default_mask_yaml).stem:
                    raise ValueError(
                        f"Strict mode mismatch: eval condition mask '{cond['mask_yaml']}' "
                        f"differs from checkpoint mask '{default_mask_yaml}'."
                    )
    else:
        grid = build_default_grid_from_training_mask(
            dataset_cfg_base=dataset_cfg_base,
            mask_cfg_base=mask_cfg_base,
            dataset_path=dataset_path,
            mask_path=mask_path,
        )

    results = []
    print(f"checkpoint={args.ckpt}  epoch={state_epoch} step={state_step}  split={args.split}")
    print(f"metric_scope={metric_scope}  report_both_metrics={report_both_metrics}")
    print(startup_summary_line(
        dataset_yaml=dataset_path,
        mask_yaml=mask_path,
        model_yaml=model_path,
        metric_scope=metric_scope,
    ))
    print(f"eval_dir={run_dir}")
    print(f"grid: {[c['name'] for c in grid]}")

    for idx, cond in enumerate(grid):
        cond_name = cond["name"]
        if "dataset_cfg" in cond and "mask_cfg" in cond:
            dataset_cfg = OmegaConf.create(OmegaConf.to_container(cond["dataset_cfg"], resolve=True))
            mask_cfg = OmegaConf.create(OmegaConf.to_container(cond["mask_cfg"], resolve=True))
        else:
            dataset_cfg, mask_cfg = load_dataset_and_mask(cond["dataset_yaml"], cond["mask_yaml"])
        loader_cfg = OmegaConf.create(OmegaConf.to_container(loader_cfg_base, resolve=True))
        apply_cli_overrides(args, dataset_cfg, loader_cfg)
        mask_cfg_cond = merge_mask_cfg(mask_cfg, cond.get("mask_ratios"), cond.get("mask_overrides"))

        dl = build_dataloader(dataset_cfg, loader_cfg, split=args.split, mask_cfg=mask_cfg_cond, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)
        mean_list = list(dataset_cfg.norm.mean)
        std_list = list(dataset_cfg.norm.std)
        mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        out = evaluate(
            model=model,
            dl=dl,
            device=device,
            loss_fn=loss_fn,
            use_amp=use_amp,
            epoch=state_epoch,
            global_step=state_step,
            logger=logger if idx == 0 else None,
            mean=mean,
            std=std,
            save_vis=args.save_vis and idx == 0,
            lpips_net=lpips_net,
            metric_scope=metric_scope,
            report_both=report_both_metrics,
        )
        metrics = {}
        for k, v in out.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            else:
                metrics[k] = v
        results.append({
            "condition": cond_name,
            "dataset_yaml": cond.get("dataset_yaml"),
            "mask_yaml": cond.get("mask_yaml"),
            "mask_ratios": cond.get("mask_ratios"),
            "mask_overrides": cond.get("mask_overrides"),
            "metrics": metrics,
        })
        print(
            f"  {cond_name}: val_loss={metrics['val_loss']:.6f} "
            f"l1={metrics.get('l1', 0):.6f} "
            f"psnr={metrics.get('psnr', 0):.4f} "
            f"ssim={metrics.get('ssim', 0):.4f} "
            f"lpips={metrics.get('lpips', 0):.4f}"
        )

    summary = {
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_name": ckpt_path.name,
        "epoch": state_epoch,
        "step": state_step,
        "split": args.split,
        "eval_dir": str(run_dir.resolve()),
        "eval_protocol": eval_profile,
        "metric_scope": metric_scope,
        "report_both_metrics": report_both_metrics,
        "eval_profile": eval_profile,
        "dataset": cfg_name(dataset_path),
        "mask": cfg_name(mask_path),
        "model": cfg_name(model_path),
        "conditions": results,
    }
    out_path = run_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")

    if logger is not None and results:
        logger.log(epoch=state_epoch, step=state_step, split=args.split, loss=results[0]["metrics"]["val_loss"], lr=0.0)


if __name__ == "__main__":
    main()
