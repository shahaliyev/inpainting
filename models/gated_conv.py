import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class GatedConv2d(nn.Module):
    """
    Core gated convolution:
      y = phi(W_f * x) * sigmoid(W_g * x)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.feat = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = _gn(out_ch) if use_norm else nn.Identity()

        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x)
        g = torch.sigmoid(self.gate(x))
        h = f * g
        h = self.norm(h)
        h = self.act(h)
        return h


class GatedResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = GatedConv2d(in_ch, out_ch, 3, 1, 1, use_norm=True, activation="silu")
        self.conv2 = GatedConv2d(out_ch, out_ch, 3, 1, 1, use_norm=True, activation="none")
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        return F.silu(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = GatedConv2d(ch, ch, 3, stride=2, padding=1, use_norm=True, activation="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = GatedConv2d(ch, ch, 3, stride=1, padding=1, use_norm=True, activation="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class GatedUNet(nn.Module):
    """
    U-Net-like gated-conv model for inpainting.
    Expected input: [B, 4, H, W] = concat(masked_rgb, mask)
    Output: [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        chs = [base_channels * int(m) for m in channel_mults]

        self.in_conv = GatedConv2d(
            in_channels, chs[0], kernel_size=3, stride=1, padding=1, use_norm=False, activation="silu"
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        cur = chs[0]
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            if i == 0:
                for _ in range(num_res_blocks):
                    blocks.append(GatedResBlock(cur, ch))
                    cur = ch
            else:
                blocks.append(GatedResBlock(cur, ch))
                cur = ch
                for _ in range(num_res_blocks - 1):
                    blocks.append(GatedResBlock(cur, cur))
            self.down_blocks.append(blocks)
            self.skip_channels.append(cur)
            if i != len(chs) - 1:
                self.downsamples.append(Downsample(cur))

        self.mid1 = GatedResBlock(cur, cur)
        self.mid2 = GatedResBlock(cur, cur)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(len(chs))):
            ch = chs[i]
            skip_ch = self.skip_channels[i]
            blocks = nn.ModuleList()

            blocks.append(GatedResBlock(cur + skip_ch, ch))
            cur = ch
            for _ in range(num_res_blocks - 1):
                blocks.append(GatedResBlock(cur, cur))

            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Upsample(cur))

        self.out_norm = _gn(cur)
        self.out_conv = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def build_gated_conv(cfg):
    channel_mults = tuple(int(x) for x in getattr(cfg, "channel_mults", [1, 2, 4, 8]))
    return GatedUNet(
        in_channels=int(getattr(cfg, "in_channels", 4)),
        out_channels=int(getattr(cfg, "out_channels", 3)),
        base_channels=int(getattr(cfg, "base_channels", 64)),
        channel_mults=channel_mults,
        num_res_blocks=int(getattr(cfg, "num_res_blocks", 2)),
    )