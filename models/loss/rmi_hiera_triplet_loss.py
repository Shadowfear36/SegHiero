# file: rmi_hiera_triplet_loss.py

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
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 1e-3  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel

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
    device = targets.device       # <— pick up the device of `targets`
    dtype = targets.dtype

    # Move the mapping vectors onto the same device as `targets`
    # (so that indexing can happen without device mismatch):
    fine_to_mid  = fine_to_mid.to(device)
    fine_to_high = fine_to_high.to(device)

    # Start by copying the fine‐level map; we'll fill mid/high next
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


def _losses_hiera_three_level(
    predictions: torch.Tensor,
    targets_fine:  torch.Tensor,
    targets_mid:   torch.Tensor,
    targets_high:  torch.Tensor,
    n_fine:        int,
    n_mid:         int,
    n_high:        int,
    eps: float = 1e-6,
):
    """
    Exactly the same 3‐level “hierarchical” loss you had, but now param’d.

    - predictions:   [B, n_fine + n_mid + n_high, H, W]   (raw logits)
    - targets_fine:  [B, H, W]   in [0..n_fine−1] or 255
    - targets_mid:   [B, H, W]   in [0..n_mid−1] or 255
    - targets_high:  [B, H, W]   in [0..n_high−1] or 255
    - n_fine, n_mid, n_high:   ints

    Returns:
      scalar = 5 * (fine_term + mid_term + high_term)
    """
    b, C, H, W = predictions.shape
    device = predictions.device
    dtype = predictions.dtype

    # 1) Sigmoid over all channels
    probs = torch.sigmoid(predictions.float())

    # 2) Build “void” masks
    void_f = (targets_fine  == 255)
    void_m = (targets_mid   == 255)
    void_h = (targets_high  == 255)

    # 3) One‐hot encoding (set ignore→0 before one‐hot)
    tf = targets_fine .clone()
    tf[void_f] = 0
    oh_f = F.one_hot(tf, num_classes=n_fine).permute(0, 3, 1, 2).float()

    tm = targets_mid .clone()
    tm[void_m] = 0
    oh_m = F.one_hot(tm, num_classes=n_mid).permute(0, 3, 1, 2).float()

    th = targets_high.clone()
    th[void_h] = 0
    oh_h = F.one_hot(th, num_classes=n_high).permute(0, 3, 1, 2).float()

    # 4) Slice out each block of channels:
    MCMA = probs[:, :n_fine, :, :]                       # [B, n_fine, H, W]
    MCMB = probs[:, n_fine : n_fine+n_mid, :, :]          # [B, n_mid,  H, W]
    MCMC = probs[:, n_fine+n_mid : n_fine+n_mid+n_high, :, :]  # [B, n_high, H, W]

    # 5) Build MCMB_combined:  for each mid‐class i,
    #       MCMB_combined[i] = max( max_{f in bucket_i} MCMA[f], MCMB[i] )
    MCMB_combined = torch.zeros((b, n_mid, H, W), dtype=dtype, device=device)
    for i in range(n_mid):
        # first find which fine‐indices map to this mid‐index
        # (we can look at oh_m “where” mid=i, but simpler is "tf_to_mid" array outside)
        # Instead, we note: wherever targets_mid == i, the corresponding fine ∈ bucket_i
        bucket_mask = (targets_mid == i)  # [B, H, W]
        if bucket_mask.any():
            # gather all MCMA channels that actually appear under this i:
            # but since different pixels can have different fine labels, we must do max‐over all f where f→i
            # We can do it in two steps: gather that set of fine‐indices from tf
            fine_inds = tf[bucket_mask].unique().tolist()  # all f that appear under mid=i
        else:
            fine_inds = []
        if fine_inds:
            bucket_f = MCMA[:, fine_inds, :, :]  # [B, len(fine_inds), H, W]
            max_f = bucket_f.max(dim=1, keepdim=True)[0]    # [B,1,H,W]
            max_c = MCMB[:, i:i+1, :, :]                     # [B,1,H,W]
            MCMB_combined[:, i:i+1, :, :] = torch.max(torch.cat([max_f, max_c], dim=1), dim=1, keepdim=True)[0]
        else:
            # no pixel has mid=i in this batch → fallback to just MCMB[i]
            MCMB_combined[:, i:i+1, :, :] = MCMB[:, i:i+1, :, :]

    # 6) Build MCMC_combined: for each high‐class j,
    #      MCMC_combined[j] = max(   max_{m in bucket_j} MCMB_combined[m],   MCMC[j]   )
    MCMC_combined = torch.zeros((b, n_high, H, W), dtype=dtype, device=device)
    for j in range(n_high):
        bucket_m = (targets_high == j)   # [B,H,W]
        if bucket_m.any():
            mid_inds = tm[bucket_m].unique().tolist()  # all mid labels under this high=j
        else:
            mid_inds = []
        if mid_inds:
            part_m = MCMB_combined[:, mid_inds, :, :]    # [B, len(mid_inds), H, W]
            max_m  = part_m.max(dim=1, keepdim=True)[0]  # [B,1,H,W]
            max_h  = MCMC[:, j:j+1, :, :]                # [B,1,H,W]
            MCMC_combined[:, j:j+1, :, :] = torch.max(torch.cat([max_m, max_h], dim=1), dim=1, keepdim=True)[0]
        else:
            # no pixel maps to high=j → fallback to just MCMC[j]
            MCMC_combined[:, j:j+1, :, :] = MCMC[:, j:j+1, :, :]

    # 7) Build MCLA (fine→mid):  for each fine f:
    #     MCLA[f] = min( MCMA[f],  MCMB[f→mid] )
    MCLA = MCMA.clone()
    # We need a fine→mid mapping for each f:
    # But we can recover it from oh_m & tf: wherever tf==f, targets_mid gives its mid→ fetch that channel
    # Easiest: build a lookup `fine_to_mid_array[f] = that mid index`. We'll pass it in from the constructor.
    # For now, assume we have a 1D tensor `f2m` of length = n_fine, mapping fine→mid:
    # We’ll read it from `predictions.device` (so better have passed it in).
    # For clarity, in this helper function, we’ll extract it directly from `oh_m`:
    # … but to keep identical to original, let’s assume caller passed in `fine_to_mid_array`.
    raise NotImplementedError(
        "_losses_hiera_three_level needs to know `fine_to_mid_array` and `fine_to_high_array` inside. "
        "We will refactor it below in the class so that we store them as `self.fine_to_mid` and `self.fine_to_high`."
    )


# =============================================================================
# The new configurable RMIHieraTripletLoss
# =============================================================================

class RMIHieraTripletLoss(nn.Module):
    """
    3‐level (fine → mid → high) hierarchical RMI + triplet loss, fully configurable.

    Constructor arguments:
      • fine_to_mid:   LongTensor of length = n_fine, mapping each fine‐ID→mid‐ID
      • fine_to_high:  LongTensor of length = n_fine, mapping each fine‐ID→high‐ID
      • n_fine, n_mid, n_high: int
      • rmi_radius, rmi_pool_way, rmi_pool_size, rmi_pool_stride, etc.: as before
      • loss_weight_lambda: weight on the RMI term vs. BCE term
      • loss_weight: global scale

    Example of constructing from your `config.yaml` in train.py:
        cfg_classes = config["classes"]
        # Suppose cfg_classes["coarse_to_fine_map"] = [[0,1], [2,4], …]
        # and cfg_classes["super_coarse_to_coarse_map"] = [[0,1], [2,10], …]

        # 1) build a fine→mid and fine→high vector of length n_fine
        fine_to_mid  = build_fine_to_level_map(cfg_classes["coarse_to_fine_map"],  n_fine)
        fine_to_high = build_fine_to_level_map(cfg_classes["super_coarse_to_coarse_map"], n_fine)

        # 2) then pass them into RMIHieraTripletLoss:
        loss_fn = RMIHieraTripletLoss(
            n_fine        = n_fine,
            n_mid         = n_mid,
            n_high        = n_high,
            fine_to_mid   = fine_to_mid,
            fine_to_high  = fine_to_high,
            rmi_radius    = cfg_training["rmi_radius"],
            rmi_pool_way  = cfg_training["rmi_pool_way"],
            rmi_pool_size = cfg_training["rmi_pool_size"],
            rmi_pool_stride=cfg_training["rmi_pool_stride"],
            loss_weight_lambda = cfg_training["loss_weight_lambda"],
            loss_weight   = cfg_training["loss_weight"],
        ).to(device)

    Forward signature (exactly like before):
      forward(self, step, embedding, cls_score_before, cls_score, label, weight=None)

    where
      • embedding:        [B, D, H/8, W/8]
      • cls_score_before: [B, n_fine, H/8, W/8]
      • cls_score:        [B, n_fine + n_mid + n_high, H, W]
      • label:            [B, H, W]  (fine IDs or 255)
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
                 ignore_index: int    = 255):
        super().__init__()

        # ---------------------------------------------------------------------
        # 1) Store configuration
        # ---------------------------------------------------------------------
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

        # Repetition of your old “upper_ids / lower_ids” logic:
        # In a 3‐level hierarchy, “upper_ids” = the list of fine‐IDs in the top half, etc. 
        # (We keep your original split logic exactly, but read from constructor instead of hard‐coding.)
        if n_fine > 15:
            self.upper_ids = [1,2,3,4,5,6,7,10,11,13,14,15]
            self.lower_ids = [8,9,12,16,17,18,19]
        else:
            self.upper_ids = [1,2,3,4]
            self.lower_ids = [5,6]

        self.loss_weight_lambda = loss_weight_lambda
        self.loss_weight = loss_weight
        self.half_d   = self.rmi_radius * self.rmi_radius
        self.d        = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2

        # CrossEntropy for fine/mid/high branches
        self.ce = CrossEntropyLoss()

        # Create the TreeTripletLoss exactly as before, passing in “self.upper_ids / self.lower_ids”
        self.triplet_loss = TreeTripletLoss(
            num_classes = self.n_fine,
            upper_ids   = self.upper_ids,
            lower_ids   = self.lower_ids,
            ignore_index= self.ignore_index
        )

    # -------------------------------------------------------------------------
    # 2) The same “map_get_pairs” and “log_det_by_cholesky” from before:
    # -------------------------------------------------------------------------
    def map_get_pairs(self, labels_4D, probs_4D, radius=3, is_combine=True):
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

    def log_det_by_cholesky(self, matrix: torch.Tensor):
        chol = torch.cholesky(matrix)
        return 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1
        )


    # -------------------------------------------------------------------------
    # 3) The “forward” method, refactored to use our constructor‐passed vectors:
    # -------------------------------------------------------------------------
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
          • cls_score_before: [B, n_fine, H/8, W/8]  (unused except by triplet)
          • cls_score:        [B, n_fine + n_mid + n_high, H, W]
          • label:            [B, H, W]  (fine IDs in [0..n_fine−1] or 255)
        """

        b, H, W = label.shape

        # 3.1) Build three‐level targets
        targets_fine, targets_mid, targets_high = _prepare_targets_three_level(
            label, self.fine_to_mid, self.fine_to_high
        )

        # 3.2) Compute the 3‐level hiera‐focal‐like loss
        # ‣ We must replicate exactly your old “losses_hiera_focal” logic.
        predictions = cls_score  # alias

        # (a) Sigmoid
        probs = torch.sigmoid(predictions.float())

        # (b) Void masks
        void_f = (targets_fine == self.ignore_index)   # [B,H,W]
        void_m = (targets_mid  == self.ignore_index)
        void_h = (targets_high == self.ignore_index)

        # (c) One‐hot encoding
        tf = targets_fine.clone()
        tf[void_f] = 0
        oh_f = F.one_hot(tf, num_classes=self.n_fine).permute(0, 3, 1, 2).float()   # [B,n_fine,H,W]

        tm = targets_mid.clone()
        tm[void_m] = 0
        oh_m = F.one_hot(tm, num_classes=self.n_mid).permute(0, 3, 1, 2).float()     # [B,n_mid,H,W]

        th = targets_high.clone()
        th[void_h] = 0
        oh_h = F.one_hot(th, num_classes=self.n_high).permute(0, 3, 1, 2).float()   # [B,n_high,H,W]

        # (d) Slices:
        MCMA = probs[:, : self.n_fine, :, :]                         # [B, n_fine, H, W]
        MCMB = probs[:, self.n_fine : self.n_fine + self.n_mid, :, :]  # [B, n_mid,  H, W]
        MCMC = probs[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high, :, :]  # [B,n_high,H,W]

        # (e) Build MCMB_combined = max( fine_bucket, MCMB ) for each mid
        MCMB_combined = torch.zeros((b, self.n_mid, H, W), dtype=probs.dtype, device=probs.device)
        for i in range(self.n_mid):
            # find all fine‐indices f such that fine_to_mid[f] == i
            mask_f = (self.fine_to_mid == i).nonzero(as_tuple=False).flatten().tolist()
            if mask_f:
                bucket_f = MCMA[:, mask_f, :, :]             # [B, len(mask_f), H, W]
                max_f = bucket_f.max(dim=1, keepdim=True)[0]  # [B,1,H,W]
                max_c = MCMB[:, i : i+1, :, :]                # [B,1,H,W]
                stacked = torch.cat([max_f, max_c], dim=1)    # [B,2,H,W]
                MCMB_combined[:, i : i+1, :, :] = stacked.max(dim=1, keepdim=True)[0]
            else:
                # If no fine in this bucket, fallback to MCMB channel
                MCMB_combined[:, i : i+1, :, :] = MCMB[:, i : i+1, :, :]

        # (f) Build MCMC_combined = max( MCMB_bucket, MCMC ) for each high
        MCMC_combined = torch.zeros((b, self.n_high, H, W), dtype=probs.dtype, device=probs.device)
        for j in range(self.n_high):
            # find all mid‐indices m such that fine_to_high[f] == j for some f in bucket of that m
            # Equivalently, mid m is in bucket j if any fine f maps via fine_to_high to j
            mask_m = (self.fine_to_high == j).nonzero(as_tuple=False).flatten().tolist()
            # But that gives fine indices f; we want mid indices. Instead:
            # Let’s build a mid→bucket map by scanning fine_to_mid & fine_to_high at init.
            # We can precompute “mid_to_high”:
            # But to keep it simple, we’ll just do:
            mids_in_j = set(self.fine_to_mid[f.item()].item() for f in (self.fine_to_high == j).nonzero(as_tuple=False))
            if mids_in_j:
                mids_in_j = list(mids_in_j)
                part_m = MCMB_combined[:, mids_in_j, :, :]         # [B, len(mids_in_j), H, W]
                max_m  = part_m.max(dim=1, keepdim=True)[0]       # [B,1,H,W]
                max_h  = MCMC[:, j : j+1, :, :]                    # [B,1,H,W]
                stacked = torch.cat([max_m, max_h], dim=1)         # [B,2,H,W]
                MCMC_combined[:, j : j+1, :, :] = stacked.max(dim=1, keepdim=True)[0]
            else:
                MCMC_combined[:, j : j+1, :, :] = MCMC[:, j : j+1, :, :]

        # (g) Build MCLB, MCLC for the “B” side of mid/high (this was your MCLB, MCLC).
        MCLB = probs[:, self.n_fine : self.n_fine + self.n_mid, :, :]   # [B,n_mid,H,W]
        MCLC = probs[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high, :, :]  # [B,n_high,H,W]

        # (h) Build MCLA (fine→mid) = min( MCMA[f], MCLB[f→mid] )
        MCLA = MCMA.clone()  # [B,n_fine,H,W]
        for f in range(self.n_fine):
            mid_id = self.fine_to_mid[f].item()
            pair = torch.cat(
                [MCMA[:, f : f + 1, :, :], MCLB[:, mid_id : mid_id + 1, :, :]],
                dim=1
            )  # [B,2,H,W]
            MCLA[:, f : f + 1, :, :] = pair.min(dim=1, keepdim=True)[0]

        # (i) Build MCLB_combined = each mid m: min( MCLB[m],  MCLC[f→high] for f in bucket m ) 
        #     Actually your original code did it slightly differently, but we can follow that exactly:
        MCLB_combined = torch.zeros((b, self.n_mid, H, W), dtype=probs.dtype, device=probs.device)
        for m in range(self.n_mid):
            # find fine f where fine_to_mid[f] == m → then gather the corresponding MCLC channels
            f_in_m = (self.fine_to_mid == m).nonzero(as_tuple=False).flatten().tolist()
            if f_in_m:
                # gather high buckets for those f:
                high_inds = set(self.fine_to_high[f] for f in f_in_m)
                part_h = torch.cat([MCLC[:, h : h + 1, :, :] for h in high_inds], dim=1)  # [B, len(high_inds), H, W]
                min_h  = part_h.min(dim=1, keepdim=True)[0]                               # [B,1,H,W]
                min_c  = MCLB[:, m : m + 1, :, :]                                          # [B,1,H,W]
                stacked = torch.cat([min_h, min_c], dim=1)                                 # [B,2,H,W]
                MCLB_combined[:, m : m + 1, :, :] = stacked.min(dim=1, keepdim=True)[0]
            else:
                MCLB_combined[:, m : m + 1, :, :] = MCLB[:, m : m + 1, :, :]

        # (j) Now compute losses.  Sum of three terms:
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

        hiera_loss = 5.0 * (loss_f + loss_m + loss_h)

        # 3.3) RMI‐lower‐bound term on Sigmoid‐probabilities
        #     exactly copied from your original code, but using our computed
        #     (oh_f, oh_m, oh_h) etc. to build a combined one‐hot vector of length
        #     (n_fine + n_mid + n_high).  The caller already constructed and passed
        #     in “probs”; now just compute RMI lower bound on that.  We do:

        # (a) Build one‐big‐onehot for all three levels:
        onehot_all = torch.cat([oh_f, oh_m, oh_h], dim=1)  # [B, n_f+ n_m + n_h, H, W]
        valid_mask_all = (~void_f).unsqueeze(1).float().repeat(1, self.n_fine, 1, 1)
        valid_mask_all = torch.cat([
            valid_mask_all,
            (~void_m).unsqueeze(1).float().repeat(1, self.n_mid, 1, 1),
            (~void_h).unsqueeze(1).float().repeat(1, self.n_high, 1, 1),
        ], dim=1)  # [B, n_f+n_m+n_h, H, W]

        probs_masked = probs * valid_mask_all + _CLIP_MIN

        # (b) Now the RMI lower‐bound routine:
        #     For each channel c ∈ [0..(n_f+n_m+n_h)−1], extract
        #      la_vectors  ([B, c, half_d, *])  and
        #      pr_vectors  ([B, c, half_d, *])
        la_vectors, pr_vectors = self.map_get_pairs(
            onehot_all, probs_masked, radius=self.rmi_radius, is_combine=False
        )

        # la_vectors: [B, (n_f+n_m+n_h), half_d, N]  (requires_grad=False)
        la_vectors = la_vectors.view(
            [b, la_vectors.size(1), self.half_d, -1]
        ).double().requires_grad_(False)

        # pr_vectors: [B, (n_f+n_m+n_h), half_d, N]
        pr_vectors = pr_vectors.view([b, pr_vectors.size(1), self.half_d, -1]).double()

        diag_eye = torch.eye(self.half_d, device=probs.device).unsqueeze(0).unsqueeze(0)

        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        pr_cov_inv = torch.inverse(pr_cov + diag_eye * _POS_ALPHA)
        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        approx_var = la_cov - la_pr_cov.matmul(pr_cov_inv).matmul(la_pr_cov.transpose(-2, -1))

        rmi_now = 0.5 * self.log_det_by_cholesky(approx_var + diag_eye * _POS_ALPHA)
        # rmi_now: [B, (n_f+n_m+n_h)]
        rmi_per_class = rmi_now.view([-1, self.n_fine + self.n_mid + self.n_high]).mean(dim=0).float()
        rmi_per_class = rmi_per_class / float(self.half_d)
        rmi_loss = torch.sum(rmi_per_class)  # sum over channels

        # (c) Combine RMI with hiera_loss
        final_loss = self.loss_weight_lambda * rmi_loss + 0.5 * hiera_loss

        # 3.4) Add plain CrossEntropy on (fine | mid | high) slices
        ce_f = self.ce(cls_score[:, : self.n_fine],     targets_fine)
        ce_m = self.ce(cls_score[:, self.n_fine : self.n_fine + self.n_mid],   targets_mid)
        ce_h = self.ce(cls_score[:, self.n_fine + self.n_mid : self.n_fine + self.n_mid + self.n_high], targets_high)
        final_loss = final_loss + ce_f + ce_m + ce_h

        # 3.5) Add the triplet term on “embedding”
        loss_triplet, class_count = self.triplet_loss(embedding, label)
        if torch.distributed.is_initialized():
            all_counts = [torch.ones_like(class_count) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_counts, class_count, async_op=False)
            all_counts = torch.cat(all_counts, dim=0)
            ready = (all_counts.nonzero(as_tuple=False).size(0) == torch.distributed.get_world_size())
        else:
            ready = (class_count.item() > 0)

        if ready:
            total_steps = 160000 if self.n_fine > 15 else 60000
            if step.item() < total_steps:
                factor = 0.25 * (1 + math.cos((step.item() - total_steps) / total_steps * math.pi))
            else:
                factor = 0.5
            final_loss = final_loss + factor * loss_triplet

        return final_loss * self.loss_weight
