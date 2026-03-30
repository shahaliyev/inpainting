import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True, act="relu"):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        x = self.bn(x)
        x = self.act(x)
        return x, m


class PartialConvUNet(nn.Module):
    
    def __init__(
        self,
        in_channels=4,       # kept for config compatibility
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        if in_channels != 4:
            raise ValueError("Expected in_channels=4 (masked_rgb + hole_mask).")

        chs = [base_channels * int(m) for m in channel_mults]

        # Encoder (first block keeps resolution, later blocks downsample)
        self.enc_blocks = nn.ModuleList()
        self.enc_blocks.append(PConvBlock(3, chs[0], k=3, s=1, p=1, use_bn=False, act="relu"))
        for i in range(1, len(chs)):
            self.enc_blocks.append(PConvBlock(chs[i - 1], chs[i], k=3, s=2, p=1, use_bn=True, act="relu"))

        # Bottleneck
        self.bottleneck = PConvBlock(chs[-1], chs[-1], k=3, s=1, p=1, use_bn=True, act="relu")

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.dec_blocks.append(
                PConvBlock(chs[i] + chs[i - 1], chs[i - 1], k=3, s=1, p=1, use_bn=True, act="leaky_relu")
            )

        # Output conv (no activation here; training code uses raw prediction)
        self.out_conv = PartialConv2d(chs[0], out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # x = [masked_rgb(3), hole_mask(1)] ; hole_mask: 1 means missing
        img = x[:, :3]
        hole_mask = x[:, 3:4].clamp(0.0, 1.0)

        # PConv uses valid-mask convention: 1 valid, 0 hole
        valid = 1.0 - hole_mask
        m = valid.expand_as(img).contiguous()
        h = img * valid

        feats = []
        masks = []

        # Encode
        for blk in self.enc_blocks:
            h, m = blk(h, m)
            feats.append(h)
            masks.append(m)

        # Bottleneck
        h, m = self.bottleneck(h, m)

        # Decode with skip connections
        # dec idx 0 consumes skip from enc stage len-2, then len-3, ...
        for i, blk in enumerate(self.dec_blocks):
            skip_h = feats[-(i + 2)]
            skip_m = masks[-(i + 2)]

            h = F.interpolate(h, size=skip_h.shape[-2:], mode="nearest")
            m = F.interpolate(m, size=skip_m.shape[-2:], mode="nearest")

            h = torch.cat([h, skip_h], dim=1)
            m = torch.cat([m, skip_m], dim=1)

            h, m = blk(h, m)

        out, _ = self.out_conv(h, m)
        return out


def build_partial_conv(cfg):
    channel_mults = tuple(int(x) for x in getattr(cfg, "channel_mults", [1, 2, 4, 8]))
    return PartialConvUNet(
        in_channels=int(getattr(cfg, "in_channels", 4)),
        out_channels=int(getattr(cfg, "out_channels", 3)),
        base_channels=int(getattr(cfg, "base_channels", 64)),
        channel_mults=channel_mults,
    )