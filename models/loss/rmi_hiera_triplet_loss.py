# file: models/loss/rmi_hiera_triplet_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .cross_entropy_loss import CrossEntropyLoss
from .rmi_tree_triplet_loss import TreeTripletLoss

# =============================================================================
# Helpers for 3‐level hierarchy (fine → mid → high)
# =============================================================================
_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6   # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0    # max clip value after softmax or sigmoid operations
_POS_ALPHA = 1e-3  # initial jitter for positive‐definiteness
_IS_SUM = 1        # sum the loss per channel


def _prepare_targets_three_level(
    targets: torch.Tensor,
    fine_to_mid: torch.Tensor,
    fine_to_high: torch.Tensor,
):
    """
    Given:
      - targets:      [B, H, W]   (fine‐labels in [0..n_fine−1] or 255=ignore)
      - fine_to_mid:  LongTensor of length n_fine, mapping each fine→mid index
      - fine_to_high: LongTensor of length n_fine, mapping each fine→high index

    Produce three target maps:
      • targets_fine:  same as `targets` (255 stays 255)
      • targets_mid:   [B, H, W] where each pixel = fine_to_mid[fine_label], or 255
      • targets_high:  [B, H, W] where each pixel = fine_to_high[fine_label], or 255

    Returns:
      (targets_fine, targets_mid, targets_high)
    """
    b, H, W = targets.shape
    device = targets.device
    dtype = targets.dtype

    # Move the mapping vectors onto the same device as `targets`
    fine_to_mid  = fine_to_mid.to(device)
    fine_to_high = fine_to_high.to(device)

    # Copy the fine‐level map; we'll fill mid/high next
    targets_fine = targets

    # Initialize mid/high maps to “ignore” (255)
    targets_mid  = torch.full((b, H, W), 255, dtype=dtype, device=device)
    targets_high = torch.full((b, H, W), 255, dtype=dtype, device=device)

    # Wherever fine != 255, propagate into mid/high by indexing
    valid_mask = (targets != 255)
    if valid_mask.any():
        fine_vals = targets[valid_mask]         # 1D tensor of fine‐labels (no 255s)
        targets_mid [valid_mask] = fine_to_mid [fine_vals]
        targets_high[valid_mask] = fine_to_high[fine_vals]

    return targets_fine, targets_mid, targets_high


# =============================================================================
# RMIHieraTripletLoss
# =============================================================================
class RMIHieraTripletLoss(nn.Module):
    """
    3‐level (fine → mid → high) hierarchical RMI + triplet loss, fully configurable.

    Constructor arguments:
      • n_fine, n_mid, n_high: ints
      • fine_to_mid:   LongTensor of length = n_fine, mapping each fine‐ID→mid‐ID
      • fine_to_high:  LongTensor of length = n_fine, mapping each fine‐ID→high‐ID
      • rmi_radius, rmi_pool_way, rmi_pool_size, rmi_pool_stride: as before
      • loss_weight_lambda: weight on the RMI term vs. BCE term
      • loss_weight: global scale (multiplies entire loss at the end)
      • ignore_index: which fine label to ignore (usually 255)

    Forward signature:
      forward(self, step, embedding, cls_score_before, cls_score, label, weight=None)

    where
      • embedding:        [B, D, H/8, W/8]
      • cls_score_before: [B, n_fine, H/8, W/8]
      • cls_score:        [B, n_fine + n_mid + n_high, H, W]
      • label:            [B, H, W] (fine IDs or 255)
    """

    def __init__(self,
                 n_fine: int,
                 n_mid: int,
                 n_high: int,
                 fine_to_mid: torch.Tensor,
                 fine_to_high: torch.Tensor,
                 rmi_radius: int      = 3,
                 rmi_pool_way: int    = 0,
                 rmi_pool_size: int   = 3,
                 rmi_pool_stride: int = 3,
                 loss_weight_lambda: float = 0.5,
                 loss_weight: float   = 1.0,
                 ignore_index: int    = 255,
                 hiera_loss_multiplier: float = 5.0,
                 triplet_margin: float = 0.6,
                 triplet_max_samples: int = 200,
                 triplet_warmup_steps_small: int = 60000,
                 triplet_warmup_steps_large: int = 160000,
                 triplet_min_factor: float = 0.25,
                 triplet_max_factor: float = 0.5):
        super().__init__()

        assert fine_to_mid.dtype == torch.long
        assert fine_to_high.dtype == torch.long
        assert fine_to_mid.numel() == n_fine
        assert fine_to_high.numel() == n_fine

        self.n_fine  = n_fine
        self.n_mid   = n_mid
        self.n_high  = n_high
        self.fine_to_mid  = fine_to_mid.clone()
        self.fine_to_high = fine_to_high.clone()
        self.ignore_index = ignore_index

        # RMI‐specific params
        self.rmi_radius    = rmi_radius
        self.rmi_pool_way  = rmi_pool_way
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        assert self.rmi_pool_size == self.rmi_pool_stride

        self.loss_weight_lambda = loss_weight_lambda
        self.loss_weight = loss_weight
        self.half_d   = self.rmi_radius * self.rmi_radius
        self.d        = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2

        # Store configurable parameters
        self.hiera_loss_multiplier = hiera_loss_multiplier
        self.triplet_warmup_steps_small = triplet_warmup_steps_small
        self.triplet_warmup_steps_large = triplet_warmup_steps_large
        self.triplet_min_factor = triplet_min_factor
        self.triplet_max_factor = triplet_max_factor

        # Plain CrossEntropy on fine/mid/high slices:
        self.ce = CrossEntropyLoss()

        # Triplet‐loss on the "embedding":
        # We now pass both mapping vectors so TreeTripletLoss knows fine→mid and fine→high
        self.triplet_loss = TreeTripletLoss(
            fine_to_mid  = self.fine_to_mid,
            fine_to_high = self.fine_to_high,
            ignore_index = self.ignore_index,
            margin = triplet_margin,
            max_samples = triplet_max_samples
        )

    def map_get_pairs(self, labels_4D, probs_4D, radius=3, is_combine=True):
        """
        Exactly as before: gather all (label, prob) patches in a radius window.
        If is_combine, we return a single stacked [B, C, radius^2 * 2, new_h, new_w].
        Otherwise we return (la_vectors, pr_vectors), each [B, C, radius^2, new_h, new_w].
        """
        label_shape = labels_4D.size()
        h, w = label_shape[2], label_shape[3]
        new_h, new_w = h - (radius - 1), w - (radius - 1)
        la_ns, pr_ns = [], []
        for y in range(0, radius):
            for x in range(0, radius):
                la_now = labels_4D[:, :, y : y + new_h, x : x + new_w]
                pr_now = probs_4D[:, :, y : y + new_h, x : x + new_w]
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            pair_ns = la_ns + pr_ns
            p_vectors = torch.stack(pair_ns, dim=2)
            return p_vectors
        else:
            la_vectors = torch.stack(la_ns, dim=2)
            pr_vectors = torch.stack(pr_ns, dim=2)
            return la_vectors, pr_vectors

    # -------------------------------------------------------------------------
    # Replace `torch.cholesky` with `torch.linalg.cholesky` and add jitter logic.
    # -------------------------------------------------------------------------
    def log_det_by_cholesky(self, matrix: torch.Tensor):
        """
        Compute log‐det(matrix) via Cholesky, but if Cholesky fails,
        add gradually larger jitter = alpha * I until PD. If still fails,
        fallback to slogdet.
        Input:
          - matrix: [..., d, d] symmetric (should be positive‐definite)
        Returns:
          - log_det: [...], i.e. log |matrix|
        """
        # Ensure exact symmetry (small numerical noise can violate it):
        mat = 0.5 * (matrix + matrix.transpose(-2, -1))

        jitter = _POS_ALPHA
        max_tries = 5
        identity = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype).expand(mat.shape[:-2] + (mat.size(-1), mat.size(-1)))

        for _ in range(max_tries):
            try:
                # torch.linalg.cholesky requires a positive‐definite matrix
                chol = torch.linalg.cholesky(mat + jitter * identity)
                # log‐det = 2 * sum(log(diagonal of chol))
                return 2.0 * torch.sum(
                    torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8),
                    dim=-1
                )
            except RuntimeError:
                jitter *= 10.0

        # If we still can’t Cholesky‐factorize, fall back to slogdet:
        mat_pd = mat + jitter * identity
        sign, ld = torch.linalg.slogdet(mat_pd)
        # slogdet may return negative sign if still not PD; assume sign>0 and take ld
        return ld

    def forward(self,
                step: torch.Tensor,
                embedding: torch.Tensor,
                cls_score_before: torch.Tensor,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight=None,
                **kwargs):
        """
        Input shapes:
          • step:             scalar Tensor (used for scheduling triplet weight)
          • embedding:        [B, D, H/8, W/8]
          • cls_score_before: [B, n_fine, H/8, W/8]  (triplet only)
          • cls_score:        [B, n_fine+n_mid+n_high, H, W]
          • label:            [B, H, W]  (fine IDs or 255=ignore)
        """

        b, H, W = label.shape

        # 1) Build three‐level targets (fine, mid, high) from the single fine mask:
        targets_fine, targets_mid, targets_high = _prepare_targets_three_level(
            label, self.fine_to_mid, self.fine_to_high
        )

        # 2) Compute the three‐level “hierarchical” loss:
        predictions = cls_score  # alias
        probs = torch.sigmoid(predictions.float())

        # Void‐masks
        void_f = (targets_fine == self.ignore_index)   # [B, H, W]
        void_m = (targets_mid  == self.ignore_index)
        void_h = (targets_high == self.ignore_index)

        # One‐hot encode (ignore→0)
        tf = targets_fine.clone()
        tf[void_f] = 0
        oh_f = F.one_hot(tf, num_classes=self.n_fine).permute(0, 3, 1, 2).float()   # [B, n_fine, H, W]

        tm = targets_mid.clone()
        tm[void_m] = 0
        oh_m = F.one_hot(tm, num_classes=self.n_mid).permute(0, 3, 1, 2).float()     # [B, n_mid, H, W]

        th = targets_high.clone()
        th[void_h] = 0
        oh_h = F.one_hot(th, num_classes=self.n_high).permute(0, 3, 1, 2).float()   # [B, n_high, H, W]

        # Slice out each block of channels:
        MCMA = probs[:, : self.n_fine, :, :]                         # [B, n_fine,   H, W]
        MCMB = probs[:, self.n_fine : self.n_fine + self.n_mid, :, :]  # [B, n_mid,    H, W]
        MCMC = probs[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high, :, :]  # [B, n_high, H, W]

        # Build MCMB_combined = max( bucket_of_fine, MCMB[channel] ) for each mid
        MCMB_combined = torch.zeros((b, self.n_mid, H, W), dtype=probs.dtype, device=probs.device)
        for i in range(self.n_mid):
            mask_f = (self.fine_to_mid == i).nonzero(as_tuple=False).flatten().tolist()
            if mask_f:
                bucket_f = MCMA[:, mask_f, :, :]             # [B, len(mask_f), H, W]
                max_f = bucket_f.max(dim=1, keepdim=True)[0]  # [B,1,H,W]
                max_c = MCMB[:, i : i+1, :, :]                # [B,1,H,W]
                stacked = torch.cat([max_f, max_c], dim=1)    # [B,2,H,W]
                MCMB_combined[:, i : i+1, :, :] = stacked.max(dim=1, keepdim=True)[0]
            else:
                MCMB_combined[:, i : i+1, :, :] = MCMB[:, i : i+1, :, :]

        # Build MCMC_combined = max( bucket_of_mid, MCMC[channel] ) for each high
        MCMC_combined = torch.zeros((b, self.n_high, H, W), dtype=probs.dtype, device=probs.device)
        for j in range(self.n_high):
            mids_in_j = set(
                self.fine_to_mid[f.item()].item()
                for f in (self.fine_to_high == j).nonzero(as_tuple=False)
            )
            if mids_in_j:
                mids_in_j = list(mids_in_j)
                part_m = MCMB_combined[:, mids_in_j, :, :]     # [B, len(mids_in_j), H, W]
                max_m  = part_m.max(dim=1, keepdim=True)[0]    # [B,1,H,W]
                max_h  = MCMC[:, j : j+1, :, :]                 # [B,1,H,W]
                stacked = torch.cat([max_m, max_h], dim=1)      # [B,2,H,W]
                MCMC_combined[:, j : j+1, :, :] = stacked.max(dim=1, keepdim=True)[0]
            else:
                MCMC_combined[:, j : j+1, :, :] = MCMC[:, j : j+1, :, :]

        # Build MCLB, MCLC for the “B” side of mid/high as in original code:
        MCLB = probs[:, self.n_fine : self.n_fine + self.n_mid, :, :]   # [B, n_mid, H,W]
        MCLC = probs[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high, :, :]  # [B, n_high, H, W]

        # Build MCLA (fine→mid) = min( MCMA[f], MCLB[f→mid] )
        MCLA = MCMA.clone()  # [B, n_fine, H, W]
        for f in range(self.n_fine):
            mid_id = self.fine_to_mid[f].item()
            pair = torch.cat(
                [MCMA[:, f : f + 1, :, :], MCLB[:, mid_id : mid_id + 1, :, :]],
                dim=1
            )  # [B,2,H,W]
            MCLA[:, f : f + 1, :, :] = pair.min(dim=1, keepdim=True)[0]

        # Build MCLB_combined = min( MCLB[m], MCLC[high_ind] for high in bucket_of_mid )
        MCLB_combined = torch.zeros((b, self.n_mid, H, W), dtype=probs.dtype, device=probs.device)
        for m in range(self.n_mid):
            f_in_m = (self.fine_to_mid == m).nonzero(as_tuple=False).flatten().tolist()
            if f_in_m:
                high_inds = set(self.fine_to_high[f] for f in f_in_m)
                part_h = torch.cat([MCLC[:, h : h + 1, :, :] for h in high_inds], dim=1)  # [B, len(high_inds), H, W]
                min_h  = part_h.min(dim=1, keepdim=True)[0]                               # [B,1,H,W]
                min_c  = MCLB[:, m : m + 1, :, :]                                          # [B,1,H,W]
                stacked = torch.cat([min_h, min_c], dim=1)                                 # [B,2,H,W]
                MCLB_combined[:, m : m + 1, :, :] = stacked.min(dim=1, keepdim=True)[0]
            else:
                MCLB_combined[:, m : m + 1, :, :] = MCLB[:, m : m + 1, :, :]

        # Now compute the three‐level “hiera‐focal‐like” BCE losses:
        valid_f = (~void_f).unsqueeze(1).float()   # [B,1,H,W]
        num_valid_f = valid_f.sum().clamp_min(1.0)
        valid_m = (~void_m).unsqueeze(1).float()
        num_valid_m = valid_m.sum().clamp_min(1.0)
        valid_h = (~void_h).unsqueeze(1).float()
        num_valid_h = valid_h.sum().clamp_min(1.0)

        loss_f = (
            (-oh_f * torch.log(MCLA + _CLIP_MIN)
             - (1 - oh_f) * torch.log(1 - MCMA + _CLIP_MIN))
            * valid_f
        ).sum() / (num_valid_f * self.n_fine)

        loss_m = (
            (-oh_m * torch.log(MCLB_combined + _CLIP_MIN)
             - (1 - oh_m) * torch.log(1 - MCMB_combined + _CLIP_MIN))
            * valid_m
        ).sum() / (num_valid_m * self.n_mid)

        loss_h = (
            (-oh_h * torch.log(MCLC + _CLIP_MIN)
             - (1 - oh_h) * torch.log(1 - MCMC_combined + _CLIP_MIN))
            * valid_h
        ).sum() / (num_valid_h * self.n_high)

        hiera_loss = self.hiera_loss_multiplier * (loss_f + loss_m + loss_h)

        # ---------------------------------------------------------------------
        # 3) RMI‐lower‐bound term on Sigmoid‐probabilities
        # ---------------------------------------------------------------------
        onehot_all = torch.cat([oh_f, oh_m, oh_h], dim=1)  # [B, n_f + n_m + n_h, H, W]
        valid_mask_all = (~void_f).unsqueeze(1).float().repeat(1, self.n_fine, 1, 1)
        valid_mask_all = torch.cat([
            valid_mask_all,
            (~void_m).unsqueeze(1).float().repeat(1, self.n_mid, 1, 1),
            (~void_h).unsqueeze(1).float().repeat(1, self.n_high, 1, 1),
        ], dim=1)  # [B, n_f+n_m+n_h, H, W]

        probs_masked = probs * valid_mask_all + _CLIP_MIN

        la_vectors, pr_vectors = self.map_get_pairs(
            onehot_all, probs_masked, radius=self.rmi_radius, is_combine=False
        )

        la_vectors = la_vectors.view([b, la_vectors.size(1), self.half_d, -1]).double().requires_grad_(False)
        pr_vectors = pr_vectors.view([b, pr_vectors.size(1), self.half_d, -1]).double()

        diag_eye = torch.eye(self.half_d, device=probs.device).unsqueeze(0).unsqueeze(0)

        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        pr_cov_inv = torch.inverse(pr_cov + diag_eye * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        approx_var = la_cov - la_pr_cov.matmul(pr_cov_inv).matmul(la_pr_cov.transpose(-2, -1))

        # Compute RMI (½ log‐det)
        rmi_now = 0.5 * self.log_det_by_cholesky(approx_var + diag_eye * _POS_ALPHA)
        rmi_per_class = rmi_now.view([-1, self.n_fine + self.n_mid + self.n_high]).mean(dim=0).float()
        rmi_per_class = rmi_per_class / float(self.half_d)
        rmi_loss = torch.sum(rmi_per_class)

        # Combine RMI with hiera‐loss
        final_loss = self.loss_weight_lambda * rmi_loss + 0.5 * hiera_loss

        # ---------------------------------------------------------------------
        # 4) Add plain CrossEntropy on (fine | mid | high) slices
        # ---------------------------------------------------------------------
        ce_f = self.ce(cls_score[:, : self.n_fine],     targets_fine)
        ce_m = self.ce(cls_score[:, self.n_fine : self.n_fine + self.n_mid],   targets_mid)
        ce_h = self.ce(cls_score[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high], targets_high)
        final_loss = final_loss + ce_f + ce_m + ce_h

        # ---------------------------------------------------------------------
        # 5) Add the triplet term on “embedding”
        # ---------------------------------------------------------------------
        loss_triplet, class_count = self.triplet_loss(embedding, label)
        if torch.distributed.is_initialized():
            all_counts = [torch.ones_like(class_count) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_counts, class_count, async_op=False)
            all_counts = torch.cat(all_counts, dim=0)
            ready = (all_counts.nonzero(as_tuple=False).size(0) == torch.distributed.get_world_size())
        else:
            ready = (class_count.item() > 0)

        if ready:
            # Use configurable warmup steps based on number of fine classes
            total_steps = self.triplet_warmup_steps_large if self.n_fine > 15 else self.triplet_warmup_steps_small
            if step.item() < total_steps:
                factor = self.triplet_min_factor * (1 + math.cos((step.item() - total_steps) / total_steps * math.pi))
            else:
                factor = self.triplet_max_factor
            final_loss = final_loss + factor * loss_triplet

        return final_loss * self.loss_weight
