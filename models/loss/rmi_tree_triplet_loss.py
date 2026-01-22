# file: rmi_tree_triplet_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeTripletLoss(nn.Module):
    """
    Tree‐structured triplet loss for a 3‐level hierarchy (fine→mid→high).

    Constructor arguments:
      • fine_to_mid:   LongTensor of length = n_fine, mapping each fine‐ID→mid‐ID
      • fine_to_high:  LongTensor of length = n_fine, mapping each fine‐ID→high‐ID
      • ignore_index:  label to ignore (default = 255)
    """

    def __init__(self,
                 fine_to_mid: torch.Tensor,
                 fine_to_high: torch.Tensor,
                 ignore_index: int = 255,
                 margin: float = 0.6,
                 max_samples: int = 200):
        super().__init__()
        self.ignore_label = ignore_index
        self.margin = margin
        self.max_samples = max_samples

        # Store the mapping vectors as buffers so they follow .to(device)
        assert fine_to_mid.dtype == torch.long
        assert fine_to_high.dtype == torch.long
        self.register_buffer('fine_to_mid',  fine_to_mid.clone())
        self.register_buffer('fine_to_high', fine_to_high.clone())

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, max_triplet: int = None):
        """
        feats:   [B, D, H_feat, W_feat]  (embedding to form triplets)
        labels:  [B, H_label, W_label]    (fine‐level ground truth in [0..n_fine−1] or ignore_index)
        """
        if max_triplet is None:
            max_triplet = self.max_samples

        device = feats.device
        B, D, Hf, Wf = feats.shape

        # 1) Resize labels to feats spatial size using nearest
        lbl = labels.unsqueeze(1).float()
        lbl = F.interpolate(lbl, size=(Hf, Wf), mode='nearest')
        lbl = lbl.squeeze(1).long()
        # Now lbl: [B, Hf, Wf]
        assert lbl.shape[1:] == feats.shape[2:], f"{lbl.shape} vs {feats.shape}"

        # 2) Flatten both feats and labels
        flat_labels = lbl.view(-1)  # [(B*Hf*Wf)]
        flat_feats  = feats.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [(B*Hf*Wf), D]

        # 3) Build per‐pixel mid and high IDs, default = -1
        mid_ids_all  = torch.full_like(flat_labels, fill_value=-1)
        high_ids_all = torch.full_like(flat_labels, fill_value=-1)

        valid_mask = (flat_labels != self.ignore_label)
        if valid_mask.any():
            valid_labels = flat_labels[valid_mask]  # fine labels for valid pixels
            mid_ids_all[valid_mask]  = self.fine_to_mid[valid_labels]
            high_ids_all[valid_mask] = self.fine_to_high[valid_labels]

        # 4) Collect existing fine‐classes (exclude ignore and background = 0)
        unique_labels = flat_labels.unique()
        exist_classes = [c.item() for c in unique_labels if (c.item() != self.ignore_label and c.item() != 0)]

        triplet_loss = 0.0
        class_count = 0

        for fine_cls in exist_classes:
            # anchor mask for this fine class
            anchor_mask = (flat_labels == fine_cls)

            # Positive mask: same mid but different fine, and not ignore
            mid_id = self.fine_to_mid[fine_cls].item()
            pos_mask = (mid_ids_all == mid_id) & (flat_labels != fine_cls) & (flat_labels != self.ignore_label)

            # Negative mask: different high, and not ignore
            high_id = self.fine_to_high[fine_cls].item()
            neg_mask = (high_ids_all != high_id) & (flat_labels != self.ignore_label)

            # Count available samples
            num_anchor = int(anchor_mask.sum().item())
            num_pos    = int(pos_mask.sum().item())
            num_neg    = int(neg_mask.sum().item())

            if num_anchor == 0 or num_pos == 0 or num_neg == 0:
                continue  # cannot form triplet for this class

            # Determine how many triplets to sample
            min_size = min(num_anchor, num_pos, num_neg, max_triplet)

            # Gather embeddings
            feats_anchor = flat_feats[anchor_mask][:min_size]  # [min_size, D]
            feats_pos    = flat_feats[pos_mask][:min_size]     # [min_size, D]
            feats_neg    = flat_feats[neg_mask][:min_size]     # [min_size, D]

            # Compute cosine‐based distances: 1 − (a⋅b)
            # shape: [min_size]
            dist_pos = 1.0 - (feats_anchor * feats_pos).sum(dim=1)
            dist_neg = 1.0 - (feats_anchor * feats_neg).sum(dim=1)

            # Use configurable margin
            margin = self.margin * torch.ones(min_size, device=device)

            # Triplet margin loss: max(0, d_pos − d_neg + margin)
            tl = dist_pos - dist_neg + margin
            tl = F.relu(tl)

            if tl.numel() > 0:
                triplet_loss += tl.mean()
                class_count += 1

        if class_count == 0:
            # No valid triplets across batch
            return None, torch.tensor([0], device=device)

        triplet_loss = triplet_loss / class_count
        return triplet_loss, torch.tensor([class_count], device=device)
