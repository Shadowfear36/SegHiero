import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    A small “projection” head that takes a feature map of shape [B, dim_in, H, W]
    and outputs a l2‐normalized embedding of dimension `proj_dim` per spatial location.
    If proj='linear' → single 1×1 conv; if proj='convmlp' → 1×1→BN→ReLU→1×1.
    """
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super().__init__()
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1, bias=False)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1, bias=False),
            )
        else:
            raise ValueError(f"Unknown proj type: {proj}")

    def forward(self, x):
        # x: [B, dim_in, H, W]
        y = self.proj(x)                # → [B, proj_dim, H, W]
        y = F.normalize(y, p=2, dim=1)  # l2‐norm along channel dim
        return y                        # [B, proj_dim, H, W]


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise‐Separable Convolution:
      (1) depthwise: groups=in_channels, kernel_size=kw, dilation=..., padding=...
      (2) pointwise: 1×1 conv to out_channels
      Each followed by BatchNorm + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False):
        super().__init__()
        # Depthwise conv (channel‐wise)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU(inplace=True)

        # Pointwise conv (1×1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        return x


class ASPPModule(nn.Module):
    """
    Standard ASPP (Atrous Spatial Pyramid Pooling) with 1×1 + three 3×3 dilated convs + image‐pool branch.
    - dilations: tuple of ints (e.g. (1,6,12,18))
    - in_channels: # input channels
    - channels:    # intermediate channels for each branch
    """
    def __init__(self, dilations, in_channels, channels):
        super().__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()

        # 1×1 conv branch (dilation=1)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ))

        # three 3×3 dilated conv branches
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))

        # image‐pool branch: Global AvgPool → 1×1 conv → upsample
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_pool_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) image‐pool branch
        imgp = self.image_pool(x)                           # [B, C, 1, 1]
        imgp = self.image_pool_conv(imgp)                   # [B, channels, 1, 1]
        imgp = F.interpolate(imgp, size=(H, W), mode='bilinear', align_corners=False)

        # 2) all other branches
        res = [imgp]
        for branch in self.branches:
            res.append(branch(x))  # each → [B, channels, H, W]

        out = torch.cat(res, dim=1)  # [B, channels*(len(dilations)+1), H, W]
        return out


class DepthwiseSeparableASPPModule(ASPPModule):
    """
    ASPP where each 3×3 branch is replaced by DepthwiseSeparableConv.
    The 1×1 branch and the image pool branch remain exactly as in ASPPModule.
    """
    def __init__(self, dilations, in_channels, channels):
        super().__init__(dilations, in_channels, channels)
        # Replace each 3×3 branch (branches[1:], i.e. dilation>1) with depthwise‐separable
        for i, d in enumerate(self.dilations[1:], start=1):
            # original branches[i] = nn.Sequential(Conv2d(dilated), BN, ReLU)
            # replace with depthwise separable
            self.branches[i] = nn.Sequential(
                DepthwiseSeparableConv(in_channels, channels, kernel_size=3,
                                        dilation=d, padding=d, bias=False)
            )
        # branches[0] (1×1) remains unchanged


class DepthwiseSeparableASPPContrastHead(nn.Module):
    """
    A PyTorch‐native “DeepLab V3+” head with:
      - Depthwise‐Separable ASPP
      - Optional C1 skip (1×1→c1_channels)
      - Two extra depthwise separable convs (“sep_bottleneck”)
      - A 1×1 “cls_seg” conv to `num_classes`
      - A ProjectionHead on the deepest feature for triplet embedding

    Usage:
        head = DepthwiseSeparableASPPContrastHead(
            in_channels=2048,       # C5 (backbone’s last feature depth)
            c1_in_channels=256,     # C1 (backbone’s early feature depth) or 0 if unused
            c1_channels=48,         # intermediate channels for C1 path
            aspp_channels=512,      # output channels of each ASPP branch
            dilations=(1,12,24,36),
            num_classes=12,         # e.g. 7 fine + 3 mid + 2 top
            proj_dim=256,           # embedding dim
            proj_type='convmlp'     # or 'linear'
        )
        logits_12, embedding = head([feat_c1, feat_c2, feat_c3, feat_c4])
    """
    def __init__(self,
                 in_channels: int,
                 c1_in_channels: int,
                 c1_channels: int,
                 aspp_channels: int,
                 dilations: tuple,
                 num_classes: int,
                 proj_dim: int = 256,
                 proj_type: str = 'convmlp'):
        super().__init__()
        # 1) Projection head (on C5 = inputs[-1])
        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj=proj_type)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

        # 2) ASPP
        #    Output of ASPPModule is: [B, aspp_channels*(len(dilations)+1), H, W]
        self.aspp = DepthwiseSeparableASPPModule(dilations=dilations,
                                                 in_channels=in_channels,
                                                 channels=aspp_channels)

        # 3) Bottleneck after ASPP
        #    The concatenated ASPP channels = (len(dilations)+1)*aspp_channels.
        total_aspp_ch = aspp_channels * (len(dilations) + 1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_aspp_ch, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        # 4) C1 skip path (optional)
        if c1_in_channels > 0:
            self.c1_bottleneck = nn.Sequential(
                nn.Conv2d(c1_in_channels, c1_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(c1_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.c1_bottleneck = None
            c1_channels = 0

        # 5) Two extra depthwise‐separable convs (“sep bottleneck”)
        #    Input channels = aspp_channels + c1_channels
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConv(aspp_channels + c1_channels, aspp_channels,
                                   kernel_size=3, padding=1, bias=False),
            DepthwiseSeparableConv(aspp_channels, aspp_channels,
                                   kernel_size=3, padding=1, bias=False),
        )

        # 6) Final segmentation head (1×1 conv → num_classes)
        self.cls_seg = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

        # 7) Alignment flags (used in interpolation)
        self.align_corners = False

    def forward(self, inputs: list):
        """
        inputs: list of four feature maps [C1, C2, C3, C4], where:
          - C1 has shape [B, c1_in_channels, H/4, W/4],
          - C4 has shape [B, in_channels,     H/8, W/8], etc.
        Returns:
          logits:    [B, num_classes, H/8, W/8]
          embedding: [B, proj_dim,      H/8, W/8]
        """
        # 1) update step counter and compute embedding from the deepest feature (C4)
        self.step += 1
        feat_c4 = inputs[-1]                       # shape [B, in_channels, H/8, W/8]
        embedding = self.proj_head(feat_c4)        # → [B, proj_dim, H/8, W/8]

        # 2) ASPP on feat_c4
        aspp_out = self.aspp(feat_c4)              # [B, aspp_ch*(#branches), H/8, W/8]
        aspp_out = self.bottleneck(aspp_out)       # → [B, aspp_ch, H/8, W/8]

        # 3) If using C1 skip, upsample ASPP to C1’s size and concat
        if self.c1_bottleneck is not None:
            feat_c1 = inputs[0]                    # [B, c1_in_channels, H/4, W/4]
            c1_processed = self.c1_bottleneck(feat_c1)  # [B, c1_channels, H/4, W/4]

            aspp_up = F.interpolate(
                aspp_out, size=c1_processed.shape[2:],
                mode='bilinear', align_corners=self.align_corners
            )  # → [B, aspp_ch, H/4, W/4]

            x = torch.cat([aspp_up, c1_processed], dim=1)  # [B, aspp_ch + c1_ch, H/4, W/4]
        else:
            x = aspp_out  # no skip

        # 4) apply two depthwise‐separable convs
        x = self.sep_bottleneck(x)  # [B, aspp_ch, H/4, W/4] if c1 used, else [B, aspp_ch, H/8, W/8]

        # 5) if we used C1, we now have x at 1/4 resolution. If no C1, x is at 1/8.
        #    The original MMCV code always upsample to C1 resolution before sep_bottleneck,
        #    so in either case the final “x” is at 1/4. But their “cls_seg” conv is applied
        #    at that 1/4 resolution and then inside the loss they downsample by 1/2 → 1/8.
        #    For simplicity here, we replicate that behavior:
        logits = self.cls_seg(x)  # [B, num_classes, H/4, W/4]

        return logits, embedding


# Example instantiation:
#
# head = DepthwiseSeparableASPPContrastHead(
#     in_channels=2048,      # C5 channels from ResNet‐101
#     c1_in_channels=256,    # C1 channels from ResNet‐101
#     c1_channels=48,
#     aspp_channels=512,
#     dilations=(1, 12, 24, 36),
#     num_classes=12,        # 7 fine + 3 mid + 2 top
#     proj_dim=256,
#     proj_type='convmlp'
# )
#
# # Later, in your model’s forward:
# feat_c1 = ...  # shape [B,256,H/4,W/4]
# feat_c2 = ...  # shape [B,512,H/8,W/8]
# feat_c3 = ...  # shape [B,1024,H/8,W/8]
# feat_c4 = ...  # shape [B,2048,H/8,W/8]
# logits_12, embedding = head([feat_c1, feat_c2, feat_c3, feat_c4])
#
# # logits_12 is [B,12,H/4,W/4].  At loss time you can downsample or upsample as needed.


