import time
import torch

from utils.metrics import compute_metrics


def masked_l1_loss(pred, target, mask, loss_fn):
    per_pix = loss_fn(pred, target)
    denom = mask.sum() * target.shape[1] + 1e-8
    return (per_pix * mask).sum() / denom


def train_one_epoch(model, dl_train, optimizer, scaler, device, loss_fn, use_amp, amp_dtype, grad_accum_steps, log_every, vis_every, epoch, global_step, logger, mean, std):
    model.train()
    running_loss = 0.0
    step_count = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dl_train, start=1):
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        masked = batch["masked_image"].to(device, non_blocking=True)
        x = torch.cat([masked, mask], dim=1)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            loss = masked_l1_loss(pred, img, mask, loss_fn) / grad_accum_steps

        scaler.scale(loss).backward()

        if batch_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        step_count += 1
        loss_value = loss.item() * grad_accum_steps
        running_loss += loss_value

        if global_step % log_every == 0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / step_count
            print(f"epoch={epoch} step={global_step} lr={lr_now:.8f} loss={avg_loss:.6f} time={dt:.1f}s")
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
    return {"train_loss": avg_epoch_loss, "num_steps": step_count, "global_step": global_step}


@torch.no_grad()
def evaluate(model, dl, device, loss_fn, use_amp, amp_dtype=None, epoch=None, global_step=None, logger=None, mean=None, std=None, save_vis=False, lpips_net=None):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    vis_saved = False
    # Image-weighted sums for full-image metrics (mean over images)
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
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
            m = compute_metrics(pred, img, mask, mean, std, lpips_net=lpips_net)
            b = pred.shape[0]
            total_psnr += m["psnr_full"] * b
            total_ssim += m["ssim_full"] * b
            if "lpips_full" in m:
                total_lpips += m["lpips_full"] * b
            total_images += b

        if logger is not None and save_vis and not vis_saved:
            recon = masked * (1.0 - mask) + pred * mask
            logger.save_val_triplet(epoch=epoch, img=img, mask=mask, recon=recon, mean=mean, std=std)
            vis_saved = True

    val_loss = total_loss / (total_weight + 1e-8)
    out = {"val_loss": val_loss, "l1_mask": val_loss}
    if compute_full_metrics and total_images > 0:
        out["psnr_full"] = total_psnr / total_images
        out["ssim_full"] = total_ssim / total_images
        if lpips_net is not None:
            out["lpips_full"] = total_lpips / total_images
    return out