# file: hiera_triplet_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .cross_entropy_loss import CrossEntropyLoss
from .tree_triplet_loss import TreeTripletLoss


def _prepare_targets_two_level(targets: torch.Tensor,
                               hiera_index: list):
    """
    Build (fine, coarse) target maps from a fine‐label tensor.
    - targets:     [B, H, W] (fine‐labels in [0..n_fine−1] or 255=ignore)
    - hiera_index: list of length = n_coarse, each item = [start, end]
                   (end is exclusive). E.g. [[0,2],[2,10],[10,24],[24,29]].

    Returns:
      • targets_fine:  same as `targets` but with ignore=255 left intact
      • targets_coarse: [B, H, W] ints in [0..n_coarse−1] or 255 for ignore
      • indices_high:   same as hiera_index, returned for clarity
    """
    b, H, W = targets.shape
    n_coarse = len(hiera_index)

    # Initialize coarse‐level target to 255 (ignore)
    targets_coarse = torch.full((b, H, W),
                                255,
                                dtype=targets.dtype,
                                device=targets.device)

    for coarse_idx, (start, end) in enumerate(hiera_index):
        # Wherever fine‐label is in [start..end−1], assign that coarse_idx
        mask = (targets >= start) & (targets < end)
        targets_coarse[mask] = coarse_idx

    return targets, targets_coarse, hiera_index


def _losses_hiera_two_level(predictions: torch.Tensor,
                            targets_fine: torch.Tensor,
                            targets_coarse: torch.Tensor,
                            n_fine: int,
                            hiera_index: list,
                            multiplier: float = 5.0,
                            eps: float = 1e-8):
    """
    Exactly the same 2‐level "hierarchical" loss you had before:
      - predictions: [B, n_fine + n_coarse, H, W] (logits before sigmoid)
      - targets_fine:   [B, H, W]  ints in [0..n_fine−1] or 255
      - targets_coarse: [B, H, W]  ints in [0..n_coarse−1] or 255
      - n_fine:       number of fine classes
      - hiera_index:  list of length = n_coarse, each = [start, end]
      - multiplier:   multiplier for the hierarchical loss term

    Returns:
      scalar loss = multiplier * (fine_term + coarse_term).
    """
    b, C, H, W = predictions.shape
    n_coarse = len(hiera_index)

    pred = torch.sigmoid(predictions.float())
    void_fine = (targets_fine == 255)
    void_coarse = (targets_coarse == 255)

    # One‐hot for fine & coarse, setting ignore→0 before one‐hot
    tf = targets_fine.clone()
    tf[void_fine] = 0
    oh_f = F.one_hot(tf, num_classes=n_fine).permute(0, 3, 1, 2)  # [B, n_fine, H, W]

    tc = targets_coarse.clone()
    tc[void_coarse] = 0
    oh_c = F.one_hot(tc, num_classes=n_coarse).permute(0, 3, 1, 2)  # [B, n_coarse, H, W]

    MCMA = pred[:, :n_fine, :, :]                         # [B, n_fine, H, W]
    MCLB = pred[:, n_fine : n_fine + n_coarse, :, :]       # [B, n_coarse, H, W]

    # Build MCMB: for each coarse bucket i, take max over {fine logits in bucket} U {that bucket’s coarse logit}
    MCMB = torch.zeros((b, n_coarse, H, W),
                       dtype=pred.dtype,
                       device=pred.device)
    for i, (start, end) in enumerate(hiera_index):
        bucket_f = MCMA[:, start:end, :, :]                # [B, end-start, H, W]
        bucket_c = MCLB[:, i : i+1, :, :]                   # [B,   1    , H, W]
        stacked = torch.cat([bucket_f, bucket_c], dim=1)    # [B, (end-start)+1, H, W]
        MCMB[:, i:i+1, :, :] = torch.max(stacked, dim=1, keepdim=True)[0]

    # Build MCLA: for each fine index f, MCLA[f] = min( MCMA[f], MCLB[coarse_of‐f] )
    MCLA = MCMA.clone()  # [B, n_fine, H, W]
    for i, (start, end) in enumerate(hiera_index):
        for fidx in range(start, end):
            pair = torch.cat([MCMA[:, fidx:fidx+1, :, :], MCLB[:, i:i+1, :, :]], dim=1)
            MCLA[:, fidx:fidx+1, :, :] = torch.min(pair, dim=1, keepdim=True)[0]

    valid_f = (~void_fine).unsqueeze(1).float()     # [B, 1, H, W]
    num_valid_f = valid_f.sum().clamp_min(1.0)
    valid_c = (~void_coarse).unsqueeze(1).float()   # [B, 1, H, W]
    num_valid_c = valid_c.sum().clamp_min(1.0)

    loss_fine = ((-oh_f * torch.log(MCLA + eps)
                  - (1 - oh_f) * torch.log(1 - MCMA + eps))
                 * valid_f).sum() / (num_valid_f * n_fine)

    loss_coarse = ((-oh_c * torch.log(MCLB + eps)
                    - (1 - oh_c) * torch.log(1 - MCMB + eps))
                   * valid_c).sum() / (num_valid_c * n_coarse)

    return multiplier * (loss_fine + loss_coarse)


class HieraTripletLoss(nn.Module):
    """
    2‐level (fine→coarse) hierarchical BCE + triplet loss.
    Expects at initialization:
      • num_classes: total number of fine‐classes  (i.e. n_fine)
      • hiera_map:   length = n_fine, mapping each fine index→coarse index
      • hiera_index: list of length = n_coarse, each = [start,end] for fine indices
      • ignore_index: which label to treat as “void” (default 255)

    In forward(), we take:
      step, embedding, cls_score_before, cls_score, label, weight=None

    where:
      • embedding:         [B, D, H/8, W/8]   (for triplet)
      • cls_score_before:  [B, n_fine, H/8, W/8]
      • cls_score:         [B, n_fine + n_coarse, H, W]
      • label:             [B, H, W]
    """

    def __init__(self,
                 num_classes: int,
                 hiera_map: list,
                 hiera_index: list,
                 ignore_index: int = 255,
                 use_sigmoid: bool = False,
                 loss_weight: float = 1.0,
                 hiera_loss_multiplier: float = 5.0,
                 triplet_margin: float = 0.6,
                 triplet_max_samples: int = 200,
                 triplet_warmup_steps: int = 80000,
                 triplet_min_factor: float = 0.25,
                 triplet_max_factor: float = 0.5):
        super().__init__()
        self.num_classes  = num_classes        # = n_fine
        self.hiera_map    = hiera_map          # length n_fine, maps each fine→coarse
        self.hiera_index  = hiera_index        # list of length n_coarse, each [start,end)
        self.ignore_index = ignore_index
        self.ce           = CrossEntropyLoss()

        # Store configurable parameters
        self.hiera_loss_multiplier = hiera_loss_multiplier
        self.triplet_warmup_steps = triplet_warmup_steps
        self.triplet_min_factor = triplet_min_factor
        self.triplet_max_factor = triplet_max_factor

        # Pass hiera_map & hiera_index into TreeTripletLoss
        self.triplet_loss_fn = TreeTripletLoss(
            num_classes=len(hiera_map),
            hiera_map=hiera_map,
            hiera_index=hiera_index,
            ignore_index=ignore_index,
            margin=triplet_margin,
            max_samples=triplet_max_samples
        )
        self.loss_weight = loss_weight

    def forward(self,
                step: torch.Tensor,
                embedding: torch.Tensor,
                cls_score_before: torch.Tensor,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight=None,
                **kwargs):
        """
        • step:              a scalar Tensor (used to schedule triplet weight)
        • embedding:         [B, D, H/8, W/8]
        • cls_score_before:  [B, n_fine, H/8, W/8]  (unused here except for triplet)
        • cls_score:         [B, n_fine + n_coarse, H, W]
        • label:             [B, H, W]  (fine class IDs in [0..n_fine−1] or 255)
        """

        # 1) Build two‐level targets
        targets_fine, targets_coarse, indices_top = _prepare_targets_two_level(
            label, self.hiera_index
        )

        # 2) 2‐level hierarchical BCE‐like loss
        loss_hiera = _losses_hiera_two_level(
            cls_score,
            targets_fine,
            targets_coarse,
            self.num_classes,
            self.hiera_index,
            multiplier=self.hiera_loss_multiplier
        )

        # 3) Add plain CrossEntropy on fine / coarse slices
        ce_loss_f = self.ce(cls_score[:, :self.num_classes], targets_fine)
        ce_loss_c = self.ce(
            cls_score[:, self.num_classes : self.num_classes + len(self.hiera_index)],
            targets_coarse
        )
        loss = loss_hiera + ce_loss_f + ce_loss_c

        # 4) Triplet on embedding (with same schedule as before)
        loss_triplet, class_count = self.triplet_loss_fn(embedding, label)
        # Gather counts across GPUs, if using DDP:
        if torch.distributed.is_initialized():
            all_counts = [torch.ones_like(class_count)
                          for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_counts, class_count, async_op=False)
            all_counts = torch.cat(all_counts, dim=0)
            ready = (all_counts.nonzero(as_tuple=False).numel() == torch.distributed.get_world_size())
        else:
            ready = (class_count.item() > 0)

        if ready:
            # Cosine schedule with configurable parameters
            if step.item() < self.triplet_warmup_steps:
                factor = self.triplet_min_factor * (1 + math.cos((step.item() - self.triplet_warmup_steps) / self.triplet_warmup_steps * math.pi))
            else:
                factor = self.triplet_max_factor
            loss = loss + factor * loss_triplet

        return loss * self.loss_weight
