import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    A lightweight, PyTorch‐only ResNet that yields the four feature stages:
      - C1: output of layer1   (stride 4)
      - C2: output of layer2   (stride 8)
      - C3: output of layer3   (stride 16)
      - C4: output of layer4   (stride 32)

    We load torchvision’s ResNet (50 or 101), strip off its avgpool/fc,
    and expose layer1→layer4.

    Usage:
        backbone = ResNetBackbone(depth=101, pretrained=True)
        c1, c2, c3, c4 = backbone(input_tensor)
        # shapes (if input is B×3×H×W):
        #   c1: [B,  256, H/4,  W/4 ]
        #   c2: [B,  512, H/8,  W/8 ]
        #   c3: [B, 1024, H/16, W/16]
        #   c4: [B, 2048, H/32, W/32]
    """
    def __init__(self, depth: int = 101, pretrained: bool = True):
        """
        Args:
          depth (int):  Either 50 or 101 (torchvision’s ResNet50/ResNet101).
          pretrained (bool):  If True, load ImageNet‐pretrained weights.
        """
        super().__init__()

        if depth == 50:
            base = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            base = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("`depth` must be 50 or 101")

        # 1) Remove avgpool & fully‐connected
        #    (“base” is a nn.Module with attributes: conv1, bn1, relu, maxpool,
        #     layer1, layer2, layer3, layer4, avgpool, fc)
        #    We will keep everything up through layer4.
        self.stem_conv  = base.conv1    # 7×7, stride=2, padding=3
        self.stem_bn    = base.bn1
        self.stem_relu  = base.relu
        self.stem_pool  = base.maxpool  # 3×3 max‐pool, stride=2

        # 2) The four “residual stages”
        self.layer1 = base.layer1  # out stride=4
        self.layer2 = base.layer2  # out stride=8
        self.layer3 = base.layer3  # out stride=16
        self.layer4 = base.layer4  # out stride=32

        # 3) We will expose feature dims:
        #    C1 channels = 256 (layer1’s output)
        #    C2 channels = 512 (layer2’s output)
        #    C3 channels = 1024 (layer3’s output)
        #    C4 channels = 2048 (layer4’s output)
        #    We do not need base.avgpool or base.fc.

    def forward(self, x: torch.Tensor):
        # Input x: [B, 3, H, W]
        x = self.stem_conv(x)            # → [B,  64, H/2,  W/2 ]
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)            # → [B,  64, H/4,  W/4 ]

        c1 = self.layer1(x)              # → [B,  256, H/4,  W/4 ]
        c2 = self.layer2(c1)             # → [B,  512, H/8,  W/8 ]
        c3 = self.layer3(c2)             # → [B, 1024, H/16, W/16]
        c4 = self.layer4(c3)             # → [B, 2048, H/32, W/32]

        return c1, c2, c3, c4
