import torch


def build_optimizer(model, optimizer_cfg):
    name = str(getattr(optimizer_cfg, "name", "adamw")).lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(optimizer_cfg, "lr", 1e-4)),
            betas=tuple(getattr(optimizer_cfg, "betas", [0.9, 0.999])),
            weight_decay=float(getattr(optimizer_cfg, "weight_decay", 0.0)),
        )

    raise ValueError(f"Unsupported optimizer.name: {name}")


def build_scheduler(optimizer, scheduler_cfg, total_epochs):
    name = str(getattr(scheduler_cfg, "name", "none")).lower()

    if name == "none":
        return None

    if name == "cosine":
        t_max = int(getattr(scheduler_cfg, "t_max", total_epochs))
        eta_min = float(getattr(scheduler_cfg, "eta_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    if name == "step":
        step_size = int(getattr(scheduler_cfg, "step_size", 30))
        gamma = float(getattr(scheduler_cfg, "gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported scheduler.name: {name}")