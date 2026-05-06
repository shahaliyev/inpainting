import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class PartialConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias
        )

        # Same geometry as input_conv, fixed all-ones kernel for mask counting.
        self.mask_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

        self.slide_winsize = float(in_ch * kernel_size * kernel_size)

    def forward(self, x, mask):
        # x: [B, C, H, W], mask: [B, C, H, W] (1 valid, 0 hole)
        x_masked = x * mask
        out = self.input_conv(x_masked)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)  # valid pixel count per output element
            no_update = mask_sum <= 0.0
            mask_ratio = self.slide_winsize / (mask_sum + 1e-8)
            mask_ratio = mask_ratio.masked_fill(no_update, 0.0)

            # Updated validity mask at output resolution/channels
            new_mask = torch.ones_like(mask_sum)
            new_mask = new_mask.masked_fill(no_update, 0.0)

        if self.input_conv.bias is not None:
            b = self.input_conv.bias.view(1, -1, 1, 1)
            out = (out - b) * mask_ratio + b
        else:
            out = out * mask_ratio

        out = out.masked_fill(no_update, 0.0)
        return out, new_mask


class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_norm=True, act="relu"):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.norm = _gn(out_ch) if use_norm else nn.Identity()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU()
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        x = self.norm(x)
        x = self.act(x)
        return x, m


class PConvResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = PConvBlock(in_ch, out_ch, k=3, s=1, p=1, use_norm=True, act="silu")
        self.conv2 = PConvBlock(out_ch, out_ch, k=3, s=1, p=1, use_norm=True, act="none")
        self.skip = None
        if in_ch != out_ch:
            self.skip = PartialConv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, m):
        h, m_h = self.conv1(x, m)
        h, m_h = self.conv2(h, m_h)
        if self.skip is None:
            s = x
        else:
            s, _ = self.skip(x, m)
        return F.silu(h + s), m_h


class PConvDownsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = PConvBlock(ch, ch, k=3, s=2, p=1, use_norm=True, act="silu")

    def forward(self, x, m):
        return self.conv(x, m)


class PConvUpsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = PConvBlock(ch, ch, k=3, s=1, p=1, use_norm=True, act="silu")

    def forward(self, x, m):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        m = F.interpolate(m, scale_factor=2, mode="nearest")
        return self.conv(x, m)


class PartialConvUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,       # kept for config compatibility
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
    ):
        super().__init__()
        if in_channels != 4:
            raise ValueError("Expected in_channels=4 (masked_rgb + hole_mask).")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")

        chs = [base_channels * int(m) for m in channel_mults]
        self.in_conv = PConvBlock(3, chs[0], k=3, s=1, p=1, use_norm=False, act="silu")

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        cur = chs[0]
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList()
            if i == 0:
                for _ in range(num_res_blocks):
                    blocks.append(PConvResBlock(cur, ch))
                    cur = ch
            else:
                blocks.append(PConvResBlock(cur, ch))
                cur = ch
                for _ in range(num_res_blocks - 1):
                    blocks.append(PConvResBlock(cur, cur))
            self.down_blocks.append(blocks)
            self.skip_channels.append(cur)
            if i != len(chs) - 1:
                self.downsamples.append(PConvDownsample(cur))

        self.mid1 = PConvResBlock(cur, cur)
        self.mid2 = PConvResBlock(cur, cur)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in reversed(range(len(chs))):
            ch = chs[i]
            skip_ch = self.skip_channels[i]
            blocks = nn.ModuleList()
            blocks.append(PConvResBlock(cur + skip_ch, ch))
            cur = ch
            for _ in range(num_res_blocks - 1):
                blocks.append(PConvResBlock(cur, cur))
            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(PConvUpsample(cur))

        self.out_conv = PartialConv2d(cur, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # x = [masked_rgb(3), hole_mask(1)] ; hole_mask: 1 means missing
        img = x[:, :3]
        hole_mask = x[:, 3:4].clamp(0.0, 1.0)

        # PConv uses valid-mask convention: 1 valid, 0 hole
        valid = 1.0 - hole_mask
        m = valid.expand_as(img).contiguous()
        h, m = self.in_conv(img * valid, valid.expand_as(img).contiguous())

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for b in blocks:
                h, m = b(h, m)
            skips.append((h, m))
            if i < len(self.downsamples):
                h, m = self.downsamples[i](h, m)

        h, m = self.mid1(h, m)
        h, m = self.mid2(h, m)

        for i, blocks in enumerate(self.up_blocks):
            skip_h, skip_m = skips.pop()
            h = torch.cat([h, skip_h], dim=1)
            m = torch.cat([m, skip_m], dim=1)
            for b in blocks:
                h, m = b(h, m)
            if i < len(self.upsamples):
                h, m = self.upsamples[i](h, m)

        out, _ = self.out_conv(h, m)
        return out


def build_partial_conv(cfg):
    channel_mults = tuple(int(x) for x in getattr(cfg, "channel_mults", [1, 2, 4, 8]))
    return PartialConvUNet(
        in_channels=int(getattr(cfg, "in_channels", 4)),
        out_channels=int(getattr(cfg, "out_channels", 3)),
        base_channels=int(getattr(cfg, "base_channels", 64)),
        channel_mults=channel_mults,
        num_res_blocks=int(getattr(cfg, "num_res_blocks", 2)),
    )