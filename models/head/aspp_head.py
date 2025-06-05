import torch
import torch.nn as nn
import torch.nn.functional as F
from .decode_head import BaseDecodeHead


class ASPPModule(nn.Module):
    """
    Pure-PyTorch Atrous Spatial Pyramid Pooling (ASPP).

    Takes an input feature map of shape [B, in_channels, H, W] and returns
    a list of feature maps (all [B, channels, H, W]) consisting of:
      1) one image-pool branch (global average → 1×1 conv → BN → ReLU → upsample)
      2) one 1×1 conv branch
      3) one 3×3 dilated conv branch for each dilation > 1
    """
    def __init__(self, dilations, in_channels, channels):
        super().__init__()
        self.dilations = dilations

        # 1) Image-pool branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 2) One branch per dilation
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                # 1×1 conv branch
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ))
            else:
                # 3×3 dilated conv branch
                self.branches.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Image-pool branch: [B, channels, 1, 1] → upsample to [B, channels, H, W]
        imgp = self.image_pool(x)
        imgp = F.interpolate(imgp, size=(H, W), mode='bilinear', align_corners=False)

        outs = [imgp]

        # 2) Each conv branch produces [B, channels, H, W]
        for branch in self.branches:
            outs.append(branch(x))

        return outs  # list of length (len(dilations)+1), each [B, channels, H, W]


class ASPPHead(BaseDecodeHead):
    """
    Pure-PyTorch ASPP-based decode head for semantic segmentation (DeepLabV3 style).

    Inherits from BaseDecodeHead, which already provides:
      - dropout → conv_seg (1×1→num_classes)
      - an `_transform_inputs` utility that picks the correct backbone feature
      - a `cls_seg(...)` method that applies dropout (if set) and then conv_seg

    This head:
      1) Runs ASPPModule on the selected feature `x`.
      2) Concatenates all ASPP outputs along the channel dimension.
      3) Feeds the concatenation into a 3×3 bottleneck conv → BN → ReLU.
      4) Calls `self.cls_seg(...)` to produce the final `num_classes` logits.
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        num_classes: int,
        dilations=(1, 6, 12, 18),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        **kwargs
    ):
        """
        Args:
          in_channels  (int):  Number of channels in the input feature (from backbone).
          channels      (int):  Number of intermediate channels inside ASPP and bottleneck.
          num_classes   (int):  Total number of output channels (fine + coarse, e.g. 29+4=33).
          dilations   (tuple):  Dilation rates for ASPP (default (1,6,12,18)).
          dropout_ratio(float):  Dropout before the final 1×1. Matches BaseDecodeHead.
          align_corners (bool):  Whether to use align_corners=False in all interpolations.
          **kwargs:          Passed into BaseDecodeHead (e.g. in_index, input_transform, etc.).
        """
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            **kwargs
        )

        # 1) ASPP
        self.aspp = ASPPModule(dilations=dilations, in_channels=in_channels, channels=channels)

        # 2) Bottleneck after concatenation:
        #    input channels = channels * (number_of_branches)
        total_branches = len(dilations) + 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * total_branches,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """
        Args:
          inputs (list[Tensor]): a list of multi-level backbone features.
                                 BaseDecodeHead._transform_inputs will pick the one at self.in_index.

        Returns:
          Tensor of shape [B, num_classes, H_out, W_out], where H_out/W_out depend on the chosen feature.
        """
        # 1) Pick the correct feature map from the backbone:
        x = self._transform_inputs(inputs)  # → [B, in_channels, H, W]

        B, C, H, W = x.shape

        # 2) Run ASPP: obtain a list of len(dilations)+1 tensors, each [B, channels, H, W]
        aspp_outs = self.aspp(x)

        # 3) Concatenate all ASPP outputs: [B, channels*(#branches), H, W]
        cat = torch.cat(aspp_outs, dim=1)

        # 4) Bottleneck conv → [B, channels, H, W]
        feats = self.bottleneck(cat)

        # 5) Final classify → [B, num_classes, H, W]
        out = self.cls_seg(feats)
        return out
