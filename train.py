import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.build import build_dataloader
from models.build import build_model
from training.checkpoint import load_checkpoint, make_checkpoint_dict, save_best_checkpoint, save_last_checkpoint, validate_checkpoint_schema
from training.engine import evaluate, train_one_epoch
from training.logger import MetricsLogger
from training.losses import build_train_loss
from training.optim import build_optimizer, build_scheduler
from utils.config_resolver import require_cfg_fields, resolve_config_path
from utils.runtime_messages import startup_summary_line
from utils.run_metadata import build_train_run_name, save_resolved_config, save_run_metadata

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Dataset config key, e.g. carpet")
    ap.add_argument("--mask", default=None, help="Mask config key, e.g. mixed")
    ap.add_argument("--model", default=None, help="Model config key, e.g. unet")
    ap.add_argument("--loader", default="default", help="Loader config key")
    ap.add_argument("--train", dest="train_name", default="default", help="Train config key")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_ckpt", default=None)
    ap.add_argument("--strict_config_match", action="store_true",
                    help="Fail instead of warn when resume CLI keys mismatch checkpoint metadata.")
    return ap.parse_args()


def apply_runtime_settings(args, train_cfg):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        if bool(getattr(train_cfg, "tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


def get_amp_dtype(train_cfg, device):
    if device.type != "cuda":
        return None
    name = str(getattr(train_cfg, "amp_dtype", "bfloat16")).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported train.amp_dtype: {name}. Use bfloat16 or float16.")


def compile_model(model, train_cfg):
    compile_cfg = getattr(train_cfg, "compile", None)
    enabled = bool(getattr(compile_cfg, "enabled", False)) if compile_cfg is not None else False
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this torch build; skipping compile")
        return model
    mode = str(getattr(compile_cfg, "mode", "default"))
    fullgraph = bool(getattr(compile_cfg, "fullgraph", False))
    dynamic = bool(getattr(compile_cfg, "dynamic", False))
    return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

def apply_thread_settings(train_cfg):
    threads_cfg = getattr(train_cfg, "threads", None)
    if threads_cfg is None:
        return
    if getattr(threads_cfg, "omp", None) is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads_cfg.omp)
    if getattr(threads_cfg, "mkl", None) is not None:
        os.environ["MKL_NUM_THREADS"] = str(threads_cfg.mkl)
    if getattr(threads_cfg, "torch", None) is not None:
        torch.set_num_threads(int(threads_cfg.torch))
    if getattr(threads_cfg, "interop", None) is not None:
        torch.set_num_interop_threads(int(threads_cfg.interop))


def apply_overrides(args, dataset_cfg, loader_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = True
        dataset_cfg.limit_seed = int(args.seed)
    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)


def load_configs(config_paths):
    dataset_cfg = OmegaConf.load(config_paths["dataset_yaml"])
    loader_cfg = OmegaConf.load(config_paths["loader_yaml"])
    mask_cfg = OmegaConf.load(config_paths["mask_yaml"])
    model_cfg = OmegaConf.load(config_paths["model_yaml"])
    train_cfg = OmegaConf.load(config_paths["train_yaml"])

    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError("dataset_cfg.root is missing")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)

    return dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg


def build_dataloaders(args, dataset_cfg, loader_cfg, mask_cfg):
    dl_train = build_dataloader(dataset_cfg, loader_cfg, split=args.split, mask_cfg=mask_cfg, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)
    dl_val = build_dataloader(dataset_cfg, loader_cfg, split=args.val_split, mask_cfg=mask_cfg, global_seed=int(args.seed), eval_seed=int(args.seed) + 1)
    return dl_train, dl_val


def main():
    args = parse_args()
    if args.resume:
        if args.resume_ckpt is None:
            raise ValueError("--resume requires --resume_ckpt pointing to an existing checkpoint.")
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_ckpt not found: {ckpt_path}")
        if ckpt_path.parent.name != "checkpoints":
            raise ValueError(f"--resume_ckpt must be inside a checkpoints directory: {ckpt_path}")
    else:
        if not args.dataset or not args.mask or not args.model:
            raise ValueError("Fresh training requires --dataset, --mask, and --model config keys.")
        ckpt_path = None

    if args.resume:
        ckpt_raw = torch.load(Path(args.resume_ckpt), map_location="cpu")
        validate_checkpoint_schema(ckpt_raw)
        config_paths = dict(ckpt_raw.get("config_paths", {}) or {})
        required = {"dataset_yaml", "loader_yaml", "mask_yaml", "model_yaml", "train_yaml"}
        if not required.issubset(config_paths.keys()):
            missing = sorted(required - set(config_paths.keys()))
            raise ValueError(f"Resume checkpoint is missing config paths: {missing}")
    else:
        config_paths = {
            "dataset_yaml": resolve_config_path("dataset", args.dataset),
            "loader_yaml": resolve_config_path("loader", args.loader),
            "mask_yaml": resolve_config_path("mask", args.mask),
            "model_yaml": resolve_config_path("model", args.model),
            "train_yaml": resolve_config_path("train", args.train_name),
        }

    dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg = load_configs(config_paths)
    require_cfg_fields(dataset_cfg, ["root", "norm.mean", "norm.std"], "dataset config")
    require_cfg_fields(loader_cfg, ["batch_size"], "loader config")
    require_cfg_fields(train_cfg, [
        "epochs",
        "grad_accum_steps",
        "mixed_precision",
        "log_every_steps",
        "vis_every_steps",
        "eval_every_epochs",
        "val_vis_every_epochs",
        "ckpt.save_last_every_epochs",
        "ckpt.save_best",
        "ckpt.patience",
        "ckpt.min_epochs",
        "ckpt.min_delta",
        "metrics.scope",
        "metrics.report_both",
        "loss.name",
        "loss.weights.l1",
        "loss.weights.perceptual",
        "loss.weights.tv",
        "optimizer",
        "scheduler",
    ], "train config")
    apply_runtime_settings(args, train_cfg)
    apply_thread_settings(train_cfg)
    apply_overrides(args, dataset_cfg, loader_cfg)

    epochs = int(train_cfg.epochs)
    max_steps_cfg = getattr(train_cfg, "max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    if max_steps is not None and max_steps <= 0:
        raise ValueError("train.max_steps must be > 0 when provided.")
    grad_accum_steps = int(train_cfg.grad_accum_steps)
    use_amp = bool(train_cfg.mixed_precision)
    log_every = int(train_cfg.log_every_steps)
    vis_every = int(train_cfg.vis_every_steps)
    eval_every_epochs = int(train_cfg.eval_every_epochs)
    eval_every_steps_cfg = getattr(train_cfg, "eval_every_steps", None)
    eval_every_steps = int(eval_every_steps_cfg) if eval_every_steps_cfg is not None else 0
    if eval_every_steps < 0:
        raise ValueError("train.eval_every_steps must be >= 0.")
    val_vis_every_epochs = int(train_cfg.val_vis_every_epochs)
    val_vis_every_steps_cfg = getattr(train_cfg, "val_vis_every_steps", None)
    val_vis_every_steps = int(val_vis_every_steps_cfg) if val_vis_every_steps_cfg is not None else 0
    if val_vis_every_steps < 0:
        raise ValueError("train.val_vis_every_steps must be >= 0.")
    save_last_every_epochs = int(train_cfg.ckpt.save_last_every_epochs)
    save_last_every_steps_cfg = getattr(train_cfg.ckpt, "save_last_every_steps", None)
    save_last_every_steps = int(save_last_every_steps_cfg) if save_last_every_steps_cfg is not None else 0
    if save_last_every_steps < 0:
        raise ValueError("train.ckpt.save_last_every_steps must be >= 0.")
    save_best = bool(train_cfg.ckpt.save_best)
    patience = int(train_cfg.ckpt.patience)
    min_epochs = int(train_cfg.ckpt.min_epochs)
    min_delta = float(train_cfg.ckpt.min_delta)
    if min_epochs < 0:
        raise ValueError("train.ckpt.min_epochs must be >= 0.")
    if min_delta < 0:
        raise ValueError("train.ckpt.min_delta must be >= 0.")
    metric_scope = str(train_cfg.metrics.scope).lower()
    report_both_metrics = bool(train_cfg.metrics.report_both)
    if metric_scope not in {"mask", "full"}:
        raise ValueError(f"Unsupported train.metrics.scope: {metric_scope}. Use 'mask' or 'full'.")

    dl_train, dl_val = build_dataloaders(args, dataset_cfg, loader_cfg, mask_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(train_cfg, device)
    model_base = build_model(model_cfg).to(device)
    optimizer = build_optimizer(model_base, train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, epochs)
    use_grad_scaler = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    train_loss_fn = build_train_loss(train_cfg, device)
    val_loss_fn = train_loss_fn.l1_fn

    if args.resume:
        checkpoint_dir = ckpt_path.parent
        run_dir = checkpoint_dir.parent
        run_name = run_dir.name
    else:
        run_name = build_train_run_name(
            model_yaml=config_paths["model_yaml"],
            dataset_yaml=config_paths["dataset_yaml"],
            mask_cfg=mask_cfg,
            seed=args.seed,
        )
        run_dir = Path(args.runs_dir) / run_name
        checkpoint_dir = run_dir / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir)
    resolved_cfg = {
        "dataset_cfg": OmegaConf.to_container(dataset_cfg, resolve=True),
        "loader_cfg": OmegaConf.to_container(loader_cfg, resolve=True),
        "mask_cfg": OmegaConf.to_container(mask_cfg, resolve=True),
        "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
    }
    if not args.resume:
        save_run_metadata(
            run_dir,
            run_name=run_name,
            seed=args.seed,
            args_dict=vars(args),
            config_paths=config_paths,
            resolved_cfg=resolved_cfg,
        )
        save_resolved_config(run_dir, resolved_cfg)
    print(f"run_dir={run_dir}")
    print(startup_summary_line(
        dataset_yaml=config_paths.get("dataset_yaml"),
        mask_yaml=config_paths.get("mask_yaml"),
        model_yaml=config_paths.get("model_yaml"),
        train_yaml=config_paths.get("train_yaml"),
        metric_scope=metric_scope,
    ))
    print(
        f"loss: name={train_loss_fn.name} "
        f"weights(l1={train_loss_fn.w_l1}, perceptual={train_loss_fn.w_perceptual}, tv={train_loss_fn.w_tv})"
    )
    print(
        f"early-stop monitor=val_loss(masked_l1) "
        f"patience={patience} eval_checks "
        f"min_epochs={min_epochs} min_delta={min_delta} "
        f"(eval_every_epochs={eval_every_epochs}, eval_every_steps={eval_every_steps})"
    )
    if max_steps is not None:
        print(f"step-based stop enabled: max_steps={max_steps}")

    mean_list = list(dataset_cfg.norm.mean)
    std_list = list(dataset_cfg.norm.std)
    mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    step = 0
    start_epoch = 1
    last_epoch_trained = 0
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(args.resume_ckpt) if args.resume_ckpt is not None else checkpoint_dir / "last.pt"

    if args.resume:
        state = load_checkpoint(path=ckpt_path, model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, device=device)
        start_epoch = int(state["epoch"]) + 1
        step = int(state["step"])
        best_val_loss = float(state["best_val_loss"])
        raw_paths = state["raw"].get("config_paths", {}) or {}
        for key, cli_val in (("dataset_yaml", args.dataset), ("mask_yaml", args.mask), ("model_yaml", args.model)):
            if cli_val:
                ckpt_stem = Path(raw_paths.get(key, "")).stem
                if ckpt_stem and ckpt_stem != cli_val:
                    msg = f"resume checkpoint {key}='{ckpt_stem}' but CLI provided '{cli_val}'. Using checkpoint config."
                    if args.strict_config_match:
                        raise ValueError(msg)
                    print(f"WARNING: {msg}")
        print(f"resumed from {ckpt_path} | epoch={start_epoch} step={step} best_val_loss={best_val_loss:.6f}")

    model = compile_model(model_base, train_cfg)

    val_loss = None

    for epoch in range(start_epoch, epochs + 1):
        step_before_epoch = step
        train_out = train_one_epoch(model=model, dl_train=dl_train, optimizer=optimizer, scaler=scaler, device=device, train_loss_fn=train_loss_fn, use_amp=use_amp, amp_dtype=amp_dtype, grad_accum_steps=grad_accum_steps, log_every=log_every, vis_every=vis_every, epoch=epoch, global_step=step, logger=logger, mean=mean, std=std, max_steps=max_steps)
        step = int(train_out["global_step"])
        last_epoch_trained = epoch
        train_loss = float(train_out["train_loss"])
        train_terms = train_out.get("loss_terms", {})
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch, step=step, split="train_epoch", loss=train_loss, lr=lr_now, terms=train_terms)
        print(f"epoch={epoch} train_loss={train_loss:.6f}")

        if int(train_out.get("num_steps", 0)) == 0 and max_steps is not None:
            print(f"reached max_steps={max_steps}; stopping training.")
            break

        eval_due_epoch = eval_every_epochs > 0 and epoch % eval_every_epochs == 0
        eval_due_steps = (
            eval_every_steps > 0
            and step > step_before_epoch
            and (step // eval_every_steps) > (step_before_epoch // eval_every_steps)
        )
        if eval_due_epoch or eval_due_steps:
            if eval_due_steps and val_vis_every_steps > 0:
                save_val_vis = step % val_vis_every_steps == 0
            else:
                save_val_vis = val_vis_every_epochs > 0 and epoch % val_vis_every_epochs == 0
            val_out = evaluate(
                model=model,
                dl=dl_val,
                device=device,
                loss_fn=val_loss_fn,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                epoch=epoch,
                global_step=step,
                logger=logger,
                mean=mean,
                std=std,
                save_vis=save_val_vis,
                metric_scope=metric_scope,
                report_both=report_both_metrics,
            )
            val_loss = float(val_out["val_loss"])
            logger.log(epoch=epoch, step=step, split="val", loss=val_loss, lr=lr_now)
            print(f"epoch={epoch} val_loss={val_loss:.6f}")

            improved = val_loss < (best_val_loss - min_delta)
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    best_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
                    best_path = save_best_checkpoint(checkpoint_dir, best_ckpt)
                    print(f"saved best checkpoint: {best_path}")
            else:
                if epoch >= min_epochs:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        print(
                            f"early stopping: no improvement larger than min_delta={min_delta} "
                            f"for {patience} eval checks after min_epochs={min_epochs} "
                            f"(best val_loss={best_val_loss:.6f})"
                        )
                        break

        if scheduler is not None:
            scheduler.step()

        save_last_due_epoch = save_last_every_epochs > 0 and epoch % save_last_every_epochs == 0
        save_last_due_steps = (
            save_last_every_steps > 0
            and step > step_before_epoch
            and (step // save_last_every_steps) > (step_before_epoch // save_last_every_steps)
        )
        if save_last_due_epoch or save_last_due_steps:
            last_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
            last_path = save_last_checkpoint(checkpoint_dir, last_ckpt)
            print(f"saved last checkpoint: {last_path}")

        if bool(train_out.get("reached_max_steps", False)):
            print(f"reached max_steps={max_steps}; stopping training.")
            break

    final_epoch = int(last_epoch_trained if last_epoch_trained > 0 else max(start_epoch - 1, 0))
    final_ckpt = make_checkpoint_dict(model=model_base, optimizer=optimizer, scheduler=scheduler, scaler=scaler if use_grad_scaler else None, model_cfg=model_cfg, dataset_cfg=dataset_cfg, loader_cfg=loader_cfg, mask_cfg=mask_cfg, train_cfg=train_cfg, config_paths=config_paths, epoch=final_epoch, step=step, seed=args.seed, best_val_loss=best_val_loss, last_val_loss=val_loss)
    final_path = save_last_checkpoint(checkpoint_dir, final_ckpt)
    print(f"done. saved: {final_path} (epoch={final_epoch})")


if __name__ == "__main__":
    main()