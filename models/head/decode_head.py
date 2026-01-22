import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod


def init_weights_normal(m):
    """
    Same behavior as mmcv.cnn.normal_init(module, mean=0, std=0.01).
    """
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def simple_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Pixel‐wise accuracy: fraction of correctly classified pixels.
    - logits: [B, C, H, W]
    - labels: [B, H, W] (integer ground‐truth, same spatial size)
    Returns a single‐element tensor = mean accuracy over all pixels in the batch.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)               # [B, H, W]
        valid = (labels != 255)                    # ignore_index = 255
        correct = (preds == labels) & valid
        acc = correct.sum().float() / valid.sum().clamp(min=1).float()
    return acc


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """
    A PyTorch‐only reimplementation of MMCV/MMSEG’s BaseDecodeHead.
    Drop‐in replacement for any subclass that expects to inherit from BaseDecodeHead.

    Args:
        in_channels (int | Sequence[int]): Number of channels for each selected feature.
            If `input_transform` is None, `in_channels` must be an int, and `in_index` an int.
            If `input_transform` is "resize_concat" or "multiple_select", then `in_channels`
            must be a list/tuple whose length matches `in_index`.
        channels (int): Number of channels _after_ any feature‐fusion but _before_ `conv_seg`.
        num_classes (int): Number of semantic classes (i.e. output channels of 1×1 conv).
        dropout_ratio (float): Dropout probability before the final `conv_seg`. Default: 0.1.
        input_transform (str|None): One of {None, "resize_concat", "multiple_select"}.
            - None: use exactly one feature map at `in_index` (scalar).  Then `in_channels` is int.
            - "resize_concat": use all feature maps at indices in `in_index` (list).  Each is
              upsampled to the size of the first one, then concatenated.  `in_channels` is the sum.
            - "multiple_select": just bundle all feature maps at `in_index` into a list. `in_channels`
              is a list of integers matching each index.
        in_index (int | Sequence[int]): If `input_transform` is None, a single int. Otherwise a
            list/tuple of ints giving which feature maps to pick from `forward(...)` input list.
        loss_decode (callable): A PyTorch loss function that takes `(logits, labels, weight=None, ignore_index=255)`
            and returns a scalar.  E.g. `nn.CrossEntropyLoss(ignore_index=255)`.  Default: `nn.CrossEntropyLoss(ignore_index=255)`.
        ignore_index (int): Which label to ignore in loss/accuracy.  Default: 255.
        sampler (callable|None): A pixel sampler function. It should have the signature
            `sampler(sample_logits, sample_labels) → sample_weights` (same shape as labels).
            If None, no sampling is done.  Default: None.
        align_corners (bool): Passed into `F.interpolate(..., align_corners=align_corners)`. Default: False.
    """

    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes,
        dropout_ratio=0.1,
        conv_cfg=None,      # ignored
        norm_cfg=None,      # ignored
        act_cfg=None,       # ignored
        in_index=-1,
        input_transform=None,
        loss_decode=None,
        ignore_index=255,
        sampler=None,
        align_corners=False
    ):
        super(BaseDecodeHead, self).__init__()

        # 1) Handle & validate inputs
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.input_transform = input_transform
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        # 2) Set up loss
        # If no loss provided, default to CrossEntropyLoss(ignore_index)
        if loss_decode is None:
            self.loss_decode = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.loss_decode = loss_decode

        # 3) Pixel sampler (if any)
        # `sampler` should be a callable: sampler(logits, labels) → weight mask
        self.sampler = sampler

        # 4) Final segmentation conv (1×1 → num_classes)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        # 5) Initialize weights of conv_seg
        self.conv_seg.apply(init_weights_normal)

    def _init_inputs(self, in_channels, in_index, input_transform):
        """
        Validates and stores:
          - self.input_transform
          - self.in_index
          - self.in_channels (single int or list of ints, or sum if resize_concat)
        """
        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"], \
                "input_transform must be one of None, 'resize_concat', 'multiple_select'"
            assert isinstance(in_channels, (list, tuple)), \
                "If input_transform != None, in_channels must be a list/tuple"
            assert isinstance(in_index, (list, tuple)), \
                "If input_transform != None, in_index must be a list/tuple"
            assert len(in_channels) == len(in_index), \
                "in_channels and in_index must have the same length when input_transform != None"
            if input_transform == "resize_concat":
                # all features will be resized → concat, so total channels = sum(in_channels)
                self.in_channels = sum(in_channels)
            else:  # "multiple_select"
                # we keep as a list; each feature’s channel is separate
                self.in_channels = list(in_channels)
        else:
            # no transform: exactly one feature, so in_channels and in_index must each be int
            assert isinstance(in_channels, int), "in_channels must be int when input_transform is None"
            assert isinstance(in_index, int), "in_index must be int when input_transform is None"
            self.in_channels = in_channels

    def extra_repr(self):
        s = f"input_transform={self.input_transform}, ignore_index={self.ignore_index}, align_corners={self.align_corners}"
        return s

    @abstractmethod
    def forward(self, inputs):
        """
        Given a list of feature maps (tensors) from the backbone, produce the “pre‐logits” tensor
        of shape [B, self.channels, H_out, W_out], or a tuple including embeddings if desired.
        The subclass must implement this and return either:
          - logits: [B, self.channels, H_out, W_out]
          - OR (logits, additional_embedding, …) if using contrastive head
        """
        raise NotImplementedError

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=None):
        """
        Standard “train” entrypoint.
        - inputs: list[Tensor], e.g. [feat0, feat1, feat2, feat3]
        - img_metas: unused here, but kept for API compatibility
        - gt_semantic_seg: [B,1,H,W] or [B,H,W] long tensor of ground‐truth labels
        - train_cfg: unused
        Returns a dict of losses, e.g. {"loss_seg": ..., "acc_seg": ...}.
        """
        # 1) Forward pass
        outputs = self.forward(inputs)

        # 2) If subclass returned a tuple (e.g. (logits, embedding)), then:
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            seg_logits = outputs[0]
        else:
            seg_logits = outputs

        # 3) Compute losses & return
        return self.losses(seg_logits, gt_semantic_seg)

    def forward_test(self, inputs, img_metas=None, test_cfg=None):
        """
        Standard “test” entrypoint. Simply calls forward(...) and returns what forward returns.
        """
        return self.forward(inputs)

    def _transform_inputs(self, inputs):
        """
        Apply `input_transform` to pick/resize/concat features.

        inputs: list of feature tensors, e.g. [feat0, feat1, feat2, feat3]

        Returns a single Tensor (if input_transform is None or "resize_concat"), or
        a list of Tensors (if input_transform is "multiple_select").
        """
        # 1) None: pick exactly one feature at index `self.in_index`
        if self.input_transform is None:
            return inputs[self.in_index]

        # 2) "resize_concat": pick each inputs[i] for i in in_index, upsample to size of first one, then concat
        if self.input_transform == "resize_concat":
            feats = [inputs[i] for i in self.in_index]
            # target size = feats[0].shape[2:]
            h0, w0 = feats[0].shape[2], feats[0].shape[3]
            ups = [
                F.interpolate(f, size=(h0, w0), mode="bilinear", align_corners=self.align_corners)
                if (f.shape[2], f.shape[3]) != (h0, w0)
                else f
                for f in feats
            ]
            return torch.cat(ups, dim=1)

        # 3) "multiple_select": just pick each inputs[i] for i in in_index, return as list
        if self.input_transform == "multiple_select":
            return [inputs[i] for i in self.in_index]

        raise ValueError(f"Unknown input_transform = {self.input_transform}")

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Final 1×1 classification layer:
        - apply Dropout2d (if any),
        - then conv_seg (1×1 → num_classes).
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    def losses(self, seg_logit: torch.Tensor, seg_label: torch.Tensor):
        """
        Compute segmentation loss + accuracy.

        - seg_logit: [B, num_classes, H_out, W_out]  (pre‐softmax)
        - seg_label: [B, 1, H, W] or [B, H, W] (long tensor, values in [0..num_classes−1] or 255)
        Returns: { "loss_seg": scalar tensor, "acc_seg": scalar tensor }.
        """
        loss_dict = {}

        # 1) If seg_label has shape [B,1,H,W], squeeze to [B,H,W]
        if seg_label.dim() == 4 and seg_label.shape[1] == 1:
            seg_label = seg_label.squeeze(1)

        # 2) Upsample seg_logit → [B, num_classes, H, W]
        B, C, Hout, Wout = seg_logit.shape
        H, W = seg_label.shape[1], seg_label.shape[2]
        if (Hout, Wout) != (H, W):
            seg_logit_full = F.interpolate(
                seg_logit, size=(H, W), mode="bilinear", align_corners=self.align_corners
            )
        else:
            seg_logit_full = seg_logit

        # 3) If a sampler is provided, get sample weights (same shape as seg_label)
        if self.sampler is not None:
            weight = self.sampler(seg_logit_full, seg_label)
        else:
            weight = None

        # 4) Compute loss
        #    Our loss_decode expects (logits, labels, weight, ignore_index)
        loss_seg = self.loss_decode(
            seg_logit_full, seg_label, weight, self.ignore_index
        )
        loss_dict["loss_seg"] = loss_seg

        # 5) Pixel accuracy
        acc = simple_accuracy(seg_logit_full, seg_label)
        loss_dict["acc_seg"] = acc

        return loss_dict
