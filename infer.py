# infer.py

import os
import argparse
import yaml
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Replace these imports with your actual model paths:
from models.backbone.resnet import ResNetBackbone
from models.head.sep_aspp_contrast_head import DepthwiseSeparableASPPContrastHead


def build_fine_to_coarse_map(coarse_to_fine_cfg: list, n_fine: int) -> torch.Tensor:
    """
    Given `coarse_to_fine_map` from YAML (a list of [start,end] or [lbl]),
    build a LongTensor of length = n_fine mapping each fine→coarse.
    """
    mapping = torch.empty(n_fine, dtype=torch.long)
    for coarse_idx, sub in enumerate(coarse_to_fine_cfg):
        if len(sub) == 1:
            lbl = int(sub[0])
            mapping[lbl] = coarse_idx
        else:
            start, end = int(sub[0]), int(sub[1])
            for f in range(start, end + 1):
                mapping[f] = coarse_idx
    return mapping


def build_hiera_index(coarse_to_fine_cfg: list) -> list:
    """
    Build “hiera_index” list of [start, end+1] for each coarse bucket.
    If YAML entry was [x] → we return [x, x+1].
    If YAML entry was [start, end] → we return [start, end+1].
    """
    hiera_index = []
    for sub in coarse_to_fine_cfg:
        if len(sub) == 1:
            lbl = int(sub[0])
            hiera_index.append([lbl, lbl + 1])
        else:
            start, end = int(sub[0]), int(sub[1])
            hiera_index.append([start, end + 1])
    return hiera_index


def build_fine_to_super_map(super_to_coarse_cfg: list, n_fine: int) -> torch.Tensor:
    """
    Similar to build_fine_to_coarse_map, but for “super_coarse_to_coarse_map”.
    """
    mapping = torch.empty(n_fine, dtype=torch.long)
    for super_idx, sub in enumerate(super_to_coarse_cfg):
        if len(sub) == 1:
            lbl = int(sub[0])
            mapping[lbl] = super_idx
        else:
            start, end = int(sub[0]), int(sub[1])
            for f in range(start, end + 1):
                mapping[f] = super_idx
    return mapping


def preprocess_image(img_path: str, resize: tuple):
    """
    Load a single image from disk, resize if requested, convert to normalized tensor.
    Returns:
      img_tensor: [1, 3, H', W'] float tensor
      orig_size:  (orig_H, orig_W)
      proc_size:  (H', W')
    """
    img = Image.open(img_path).convert("RGB")
    orig_W, orig_H = img.size

    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)
        proc_W, proc_H = resize
    else:
        proc_W, proc_H = orig_W, orig_H

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    img_t = to_tensor(img)
    img_t = normalize(img_t)
    img_t = img_t.unsqueeze(0)  # [1,3,H',W']
    return img_t, (orig_H, orig_W), (proc_H, proc_W)


def save_mask(mask: np.ndarray, save_path: str):
    """
    Save a 2D numpy array (H×W) of integer class IDs as a PNG.
    """
    im = Image.fromarray(mask.astype(np.uint8))
    im.save(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single image using a trained SegHiero model and YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth checkpoint. Overrides config if provided",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cpu' or 'cuda:0'); defaults to config['training']['device']",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save predicted masks (default: current directory)",
    )
    args = parser.parse_args()

    # 1) Load config.yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg["training"]["device"])

    # 2) Read hierarchy information
    n_fine = len(cfg["classes"]["fine_names"])
    n_coarse = len(cfg["classes"]["coarse_names"])
    has_super = "super_coarse_names" in cfg["classes"]
    n_super = len(cfg["classes"].get("super_coarse_names", {}))

    if has_super:
        total_classes = n_fine + n_coarse + n_super
    else:
        total_classes = n_fine + n_coarse

    # Build mapping tensors (for labels if needed)
    if has_super:
        coarse_to_fine_cfg = cfg["classes"]["coarse_to_fine_map"]
        super_to_coarse_cfg = cfg["classes"]["super_coarse_to_coarse_map"]
        fine_to_coarse = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine).to(device)
        fine_to_super = build_fine_to_super_map(super_to_coarse_cfg, n_fine).to(device)
    else:
        coarse_to_fine_cfg = cfg["classes"]["coarse_to_fine_map"]
        fine_to_coarse = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine).to(device)

    # 3) Build model architecture
    backbone = ResNetBackbone(depth=101, pretrained=False).to(device)
    aspp_head = DepthwiseSeparableASPPContrastHead(
        in_channels=2048,
        c1_in_channels=256,
        c1_channels=48,
        aspp_channels=512,
        dilations=(1, 12, 24, 36),
        num_classes=total_classes,
        proj_dim=256,
        proj_type="convmlp",
    ).to(device)
    backbone.eval()
    aspp_head.eval()

    # 4) Load checkpoint (either from CLI or config)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir = cfg["output"]["checkpoint_dir"]
        project_name = cfg["output"]["project_name"]
        ckpt_path = os.path.join(ckpt_dir, f"{project_name}_best.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone.load_state_dict(ckpt["backbone_state_dict"])
    aspp_head.load_state_dict(ckpt["aspp_head_state_dict"])

    # 5) Preprocess input image
    resize_cfg = cfg.get("transform", {}).get("resize", None)
    if resize_cfg is not None:
        resize = (int(resize_cfg[0]), int(resize_cfg[1]))
    else:
        resize = None

    img_t, (orig_H, orig_W), (proc_H, proc_W) = preprocess_image(args.image, resize)
    img_t = img_t.to(device)

    # 6) Forward pass
    with torch.no_grad():
        c1, c2, c3, c4 = backbone(img_t)
        logits, embedding = aspp_head([c1, c2, c3, c4])
        # logits: [1, total_classes, H/4, W/4]

        # Upsample logits to full-size image
        logits_full = F.interpolate(
            logits, size=(orig_H, orig_W), mode="bilinear", align_corners=False
        )  # [1, total_classes, orig_H, orig_W]

        # Split into fine / coarse / super channels
        fine_logits = logits_full[:, :n_fine, :, :]                      # [1, n_fine, H, W]
        coarse_logits = logits_full[:, n_fine : n_fine + n_coarse, :, :]  # [1, n_coarse, H, W]
        if has_super:
            super_logits = logits_full[:, n_fine + n_coarse :, :, :]      # [1, n_super, H, W]

        # Compute argmax predictions
        fine_pred = fine_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)    # [H, W]
        coarse_pred = coarse_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if has_super:
            super_pred = super_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 7) Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    fine_out_path = os.path.join(args.output_dir, f"{base_name}_fine.png")
    save_mask(fine_pred, fine_out_path)
    print(f"→ Saved fine‐level mask to {fine_out_path}")

    coarse_out_path = os.path.join(args.output_dir, f"{base_name}_coarse.png")
    save_mask(coarse_pred, coarse_out_path)
    print(f"→ Saved coarse‐level mask to {coarse_out_path}")

    if has_super:
        super_out_path = os.path.join(args.output_dir, f"{base_name}_super.png")
        save_mask(super_pred, super_out_path)
        print(f"→ Saved super‐level mask to {super_out_path}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
