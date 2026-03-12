import torch
from torchvision.utils import make_grid, save_image


def denorm(x, mean, std):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def save_triplet(path, img, mask, recon, mean, std, nrow=3):
    img_v = denorm(img, mean, std)
    recon_v = denorm(recon, mean, std)
    mask3 = mask.expand_as(img_v)
    grid = make_grid(torch.cat([img_v, mask3, recon_v], dim=0), nrow=nrow)
    save_image(grid, path)
