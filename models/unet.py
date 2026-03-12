import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Halves spatial dimensions (H, W) using a strided 3x3 convolution."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
    ):
        super().__init__()

        chs = [base_channels * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        cur = chs[0]
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            if i == 0:
                for _ in range(num_res_blocks):
                    blocks.append(ResBlock(cur, ch))
                    cur = ch
            else:
                blocks.append(ResBlock(cur, ch))
                cur = ch
                for _ in range(num_res_blocks - 1):
                    blocks.append(ResBlock(cur, cur))
            self.down_blocks.append(blocks)
            self.skip_channels.append(cur)
            if i != len(chs) - 1:
                self.downsamples.append(Downsample(cur))

        self.mid1 = ResBlock(cur, cur)
        self.mid2 = ResBlock(cur, cur)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(len(chs))):
            ch = chs[i]
            blocks = nn.ModuleList()
            skip_ch = self.skip_channels[i]

            blocks.append(ResBlock(cur + skip_ch, ch))
            cur = ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(cur, cur))

            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Upsample(cur))

        self.out_norm = _gn(cur)
        self.out_conv = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for b in blocks:
                h = b(h)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h)
        h = self.mid2(h)

        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for b in blocks:
                h = b(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


def build_unet(cfg):
    channel_mults = tuple(int(x) for x in cfg.channel_mults)
    return UNet(
        in_channels=int(cfg.in_channels),
        out_channels=int(cfg.out_channels),
        base_channels=int(cfg.base_channels),
        channel_mults=channel_mults,
        num_res_blocks=int(cfg.num_res_blocks),
    )