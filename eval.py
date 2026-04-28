import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf

from data.build import build_dataloader
from models.build import build_model
from training.engine import evaluate
from training.logger import MetricsLogger

CONFIGS_DIR = Path("configs")


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
    return ap.parse_args()


def _cfg_from_ckpt_raw(ckpt_raw, key):
    if key not in ckpt_raw or ckpt_raw[key] is None:
        raise ValueError(f"Checkpoint is missing required '{key}'. Re-train with latest train.py.")
    return OmegaConf.create(ckpt_raw[key])


def resolve_config_path(group: str, name: str) -> str:
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown {group} config '{name}': {path}")
    return str(path)


def apply_cli_overrides(args, dataset_cfg, loader_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = False
        dataset_cfg.limit_seed = int(args.seed)
    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)


def _ensure_list(x, default=None):
    if x is None:
        return default if default is not None else []
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


def _expand_grid_product(grid_cfg, defaults_cfg, default_dataset_yaml, default_mask_yaml):
    datasets = _ensure_list(getattr(grid_cfg, "dataset_yaml", None), [default_dataset_yaml])
    ratios_list = _ensure_list(getattr(grid_cfg, "mask_ratios", None))
    default_mask = getattr(defaults_cfg, "mask_yaml", default_mask_yaml)
    mask_yamls = _ensure_list(getattr(grid_cfg, "mask_yaml", None), [default_mask])
    add_mixed = bool(getattr(grid_cfg, "add_mixed", False))
    conditions = []

    ratios_for_product = ratios_list if ratios_list else [None]
    for dataset_yaml, ratio, mask_yaml in itertools.product(datasets, ratios_for_product, mask_yamls):
        name_parts = [Path(dataset_yaml).stem]
        if len(mask_yamls) > 1:
            name_parts.append(Path(mask_yaml).stem)
        if ratio is not None:
            name_parts.append(f"ratio_{ratio}")
        cond = {
            "name": "_".join(name_parts) if name_parts else "default",
            "dataset_yaml": dataset_yaml,
            "mask_yaml": mask_yaml,
            "mask_overrides": None,
        }
        if ratio is not None:
            cond["mask_ratios"] = [int(ratio)]
        conditions.append(cond)

    if add_mixed and ratios_list:
        for dataset_yaml, mask_yaml in itertools.product(datasets, mask_yamls):
            name_parts = [Path(dataset_yaml).stem]
            if len(mask_yamls) > 1:
                name_parts.append(Path(mask_yaml).stem)
            name_parts.append("mixed")
            conditions.append({
                "name": "_".join(name_parts),
                "dataset_yaml": dataset_yaml,
                "mask_yaml": mask_yaml,
                "mask_ratios": [int(r) for r in ratios_list],
                "mask_overrides": None,
            })
    return conditions


def get_eval_grid(eval_cfg, default_dataset_yaml, default_mask_yaml):
    conditions = []

    for i, c in enumerate(list(getattr(eval_cfg, "conditions", None) or [])):
        raw_overrides = getattr(c, "mask_overrides", None)
        overrides = dict(OmegaConf.to_container(raw_overrides)) if raw_overrides else None
        conditions.append({
            "name": getattr(c, "name", f"cond_{i}"),
            "dataset_yaml": getattr(c, "dataset_yaml", default_dataset_yaml),
            "mask_yaml": getattr(c, "mask_yaml", default_mask_yaml),
            "mask_ratios": _ensure_list(getattr(c, "mask_ratios", None)) or None,
            "mask_overrides": overrides,
        })

    grid_cfg = getattr(eval_cfg, "grid", None)
    if grid_cfg is not None:
        defaults_cfg = getattr(eval_cfg, "defaults", OmegaConf.create({}))
        conditions.extend(_expand_grid_product(grid_cfg, defaults_cfg, default_dataset_yaml, default_mask_yaml))

    if conditions:
        return conditions

    ratios = _ensure_list(getattr(eval_cfg, "mask_ratios", None))
    if not ratios:
        return [{
            "name": "default",
            "dataset_yaml": default_dataset_yaml,
            "mask_yaml": default_mask_yaml,
            "mask_ratios": None,
            "mask_overrides": None,
        }]
    return [{
        "name": f"ratio_{r}",
        "dataset_yaml": default_dataset_yaml,
        "mask_yaml": default_mask_yaml,
        "mask_ratios": [int(r)],
        "mask_overrides": None,
    } for r in ratios]


def load_dataset_and_mask(dataset_yaml, mask_yaml):
    dataset_cfg = OmegaConf.load(dataset_yaml)
    mask_cfg = OmegaConf.load(mask_yaml)
    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError(f"dataset_cfg.root is missing in {dataset_yaml}")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)
    return dataset_cfg, mask_cfg


def merge_mask_cfg(base_mask_cfg, mask_ratios=None, mask_overrides=None):
    if not mask_ratios and not mask_overrides:
        return base_mask_cfg
    cfg = OmegaConf.create(OmegaConf.to_container(base_mask_cfg, resolve=True))
    if mask_ratios:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"ratios": [int(r) for r in mask_ratios]}))
    if mask_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(mask_overrides)))
    return cfg


def infer_eval_dir_from_ckpt(ckpt_path: Path, eval_profile: str, split: str, epoch: int) -> Path:
    if ckpt_path.parent.name != "checkpoints":
        raise ValueError(f"Checkpoint path must be inside a 'checkpoints' directory: {ckpt_path}")
    train_run_dir = ckpt_path.parent.parent
    return train_run_dir / "eval" / eval_profile / split / f"epoch_{int(epoch)}"


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

    if eval_yaml:
        eval_cfg = OmegaConf.load(eval_yaml)
        default_dataset_yaml = dataset_path
        default_mask_yaml = mask_path
        if not default_dataset_yaml or not default_mask_yaml:
            raise ValueError("Checkpoint metadata missing config_paths.dataset_yaml/mask_yaml for eval grid usage.")
        grid = get_eval_grid(eval_cfg, default_dataset_yaml, default_mask_yaml)
    else:
        grid = [{
            "name": "default",
            "dataset_cfg": dataset_cfg_base,
            "mask_cfg": mask_cfg_base,
            "mask_ratios": None,
            "mask_overrides": None,
            "dataset_yaml": None,
            "mask_yaml": None,
        }]

    results = []
    print(f"checkpoint={args.ckpt}  epoch={state_epoch} step={state_step}  split={args.split}")
    print(f"metric_scope={metric_scope}  report_both_metrics={report_both_metrics}")
    dataset_name = Path(str(dataset_path)).stem if dataset_path else "unknown"
    mask_name = Path(str(mask_path)).stem if mask_path else "unknown"
    model_name = Path(str(model_path)).stem if model_path else "unknown"
    print(f"startup: dataset={dataset_name} mask={mask_name} model={model_name}")
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
        "epoch": state_epoch,
        "step": state_step,
        "split": args.split,
        "eval_dir": str(run_dir.resolve()),
        "metric_scope": metric_scope,
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
