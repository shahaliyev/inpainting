import time
import torch

from utils.metrics import compute_metrics


def train_one_epoch(model, dl_train, optimizer, scaler, device, train_loss_fn, use_amp, amp_dtype, grad_accum_steps, log_every, vis_every, epoch, global_step, logger, mean, std):
    model.train()
    running_loss = 0.0
    step_count = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    running_term_sums = {}

    for batch_idx, batch in enumerate(dl_train, start=1):
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        masked = batch["masked_image"].to(device, non_blocking=True)
        x = torch.cat([masked, mask], dim=1)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss, loss_terms = train_loss_fn(pred, img, mask, mean, std)
            loss = total_loss / grad_accum_steps

        scaler.scale(loss).backward()

        if batch_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        step_count += 1
        loss_value = loss.item() * grad_accum_steps
        running_loss += loss_value
        for k, v in loss_terms.items():
            running_term_sums[k] = running_term_sums.get(k, 0.0) + float(v.item())

        if global_step % log_every == 0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / step_count
            parts = [f"{k}={running_term_sums[k] / step_count:.6f}" for k in sorted(running_term_sums.keys())]
            detail = " ".join(parts)
            print(f"epoch={epoch} step={global_step} lr={lr_now:.8f} loss={avg_loss:.6f} {detail} time={dt:.1f}s")
            if logger is not None:
                logger.log(epoch=epoch, step=global_step, split="train", loss=avg_loss, lr=lr_now)

        if logger is not None and vis_every > 0 and global_step % vis_every == 0:
            with torch.no_grad():
                recon = masked * (1.0 - mask) + pred * mask
            logger.save_train_triplet(step=global_step, img=img, mask=mask, recon=recon, mean=mean, std=std)

    if len(dl_train) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_epoch_loss = running_loss / max(step_count, 1)
    avg_terms = {k: (v / max(step_count, 1)) for k, v in running_term_sums.items()}
    return {
        "train_loss": avg_epoch_loss,
        "loss_terms": avg_terms,
        "num_steps": step_count,
        "global_step": global_step,
    }


@torch.no_grad()
def evaluate(
    model,
    dl,
    device,
    loss_fn,
    use_amp,
    amp_dtype=None,
    epoch=None,
    global_step=None,
    logger=None,
    mean=None,
    std=None,
    save_vis=False,
    lpips_net=None,
    metric_scope="mask",
    report_both=True,
):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    vis_saved = False
    metric_sums = {}
    total_images = 0
    compute_full_metrics = mean is not None and std is not None

    for batch in dl:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        masked = batch["masked_image"].to(device, non_blocking=True)
        x = torch.cat([masked, mask], dim=1)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            loss_num = (loss_fn(pred, img) * mask).sum()

        total_loss += loss_num.item()
        total_weight += mask.sum().item() * img.shape[1]

        if compute_full_metrics:
            m = compute_metrics(
                pred,
                img,
                mask,
                mean,
                std,
                lpips_net=lpips_net,
                metric_scope=metric_scope,
                report_both=report_both,
            )
            b = pred.shape[0]
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * b
            total_images += b

        if logger is not None and save_vis and not vis_saved:
            recon = masked * (1.0 - mask) + pred * mask
            logger.save_val_triplet(epoch=epoch, img=img, mask=mask, recon=recon, mean=mean, std=std)
            vis_saved = True

    val_loss = total_loss / (total_weight + 1e-8)
    out = {"val_loss": val_loss}
    if compute_full_metrics and total_images > 0:
        for k, v in metric_sums.items():
            out[k] = v / total_images
    out["metric_scope"] = str(metric_scope).lower()
    return out