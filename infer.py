# infer.py

import os
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont
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
    build a LongTensor of length = n_fine mapping each fine→coarse index.
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
    Load an image, resize if requested, convert to normalized tensor.

    Returns:
      img_tensor: [1, 3, H', W']
      orig_size:  (orig_H, orig_W)
      proc_size:  (H', W')
      orig_pil:   original PIL RGB image
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
    return img_t, (orig_H, orig_W), (proc_H, proc_W), img


def save_mask(mask: np.ndarray, save_path: str):
    """
    Save a 2D numpy array (H×W) of integer class IDs as a PNG.
    """
    im = Image.fromarray(mask.astype(np.uint8))
    im.save(save_path)


def create_colormap(n: int):
    """
    Create a simple color map of n distinct RGB values.
    """
    base_colors = [
        (128, 64, 128),   # purple-ish
        (244, 35, 232),   # pink
        (70, 70, 70),     # dark gray
        (102, 102, 156),  # lavender
        (190, 153, 153),  # pale pink
        (153, 153, 153),  # light gray
        (250, 170, 30),   # orange
        (220, 220, 0),    # yellow
        (107, 142, 35),   # olive
        (152, 251, 152),  # pale green
        (70, 130, 180),   # steel blue
        (220, 20, 60),    # crimson
        (255, 0, 0),      # red
        (0, 0, 142),      # navy
        (0, 0, 70),       # dark blue
        (0, 60, 100),     # slate
        (0, 80, 100),     # teal
        (0, 0, 230),      # bright blue
        (119, 11, 32),    # maroon
    ]
    cmap = []
    for i in range(n):
        cmap.append(base_colors[i % len(base_colors)])
    return cmap


def mask_to_color_image(mask: np.ndarray, colormap: list) -> Image.Image:
    """
    Convert a 2D numpy mask (H×W) to a full-color PIL image using colormap.
    """
    H, W = mask.shape
    color_img = Image.new("RGB", (W, H))
    pixels = color_img.load()
    for y in range(H):
        for x in range(W):
            class_id = int(mask[y, x])
            if class_id < 0:
                pixels[x, y] = (0, 0, 0)
            else:
                pixels[x, y] = colormap[class_id]
    return color_img


def draw_class_indices(mask: np.ndarray,
                       base_img: Image.Image,
                       font_path: str = None) -> Image.Image:
    """
    Draw each class index number at the centroid of its region in the mask.
    - mask:     2D numpy array (H×W) of class IDs
    - base_img: PIL RGB image on which to draw indices (should match mask size)

    Uses draw.textbbox(...) to measure text dimensions.
    """
    H, W = mask.shape
    result = base_img.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", size=max(12, W // 100))
    except Exception:
        font = ImageFont.load_default()

    class_ids = np.unique(mask)
    for class_id in class_ids:
        if class_id < 0:
            continue
        ys, xs = np.where(mask == class_id)
        if len(xs) == 0:
            continue
        centroid_x = int(xs.mean())
        centroid_y = int(ys.mean())
        text = str(class_id)

        # Use draw.textbbox to compute width/height:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_position = (centroid_x - text_w // 2, centroid_y - text_h // 2)

        # Draw black outline
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((text_position[0] + dx, text_position[1] + dy),
                      text, font=font, fill="black")
        # Draw white text
        draw.text(text_position, text, fill="white", font=font)

    return result


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
        help="Directory to save predicted masks + colored overlays (default: current directory)",
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

    # 2) Read class-name dictionaries
    fine_names_dict   = {int(k): v for k, v in cfg["classes"]["fine_names"].items()}
    coarse_names_dict = {int(k): v for k, v in cfg["classes"]["coarse_names"].items()}
    has_super         = "super_coarse_names" in cfg["classes"]
    super_names_dict  = {}
    if has_super:
        super_names_dict = {int(k): v for k, v in cfg["classes"]["super_coarse_names"].items()}

    n_fine   = len(fine_names_dict)
    n_coarse = len(coarse_names_dict)
    n_super  = len(super_names_dict) if has_super else 0

    if has_super:
        total_classes = n_fine + n_coarse + n_super
    else:
        total_classes = n_fine + n_coarse

    # 3) Build mapping tensors (not used directly for drawing, but consistent with training)
    if has_super:
        coarse_to_fine_cfg  = cfg["classes"]["coarse_to_fine_map"]
        super_to_coarse_cfg = cfg["classes"]["super_coarse_to_coarse_map"]
        _ = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine).to(device)
        _ = build_fine_to_super_map(super_to_coarse_cfg, n_fine).to(device)
    else:
        coarse_to_fine_cfg = cfg["classes"]["coarse_to_fine_map"]
        _ = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine).to(device)

    # 4) Build model architecture
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

    # 5) Load checkpoint (either CLI or config default)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir     = cfg["output"]["checkpoint_dir"]
        project_name = cfg["output"]["project_name"]
        ckpt_path    = os.path.join(ckpt_dir, f"{project_name}_best.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    aspp_head.load_state_dict(ckpt["aspp_head_state_dict"])

    # 6) Preprocess input image
    resize_cfg = cfg.get("transform", {}).get("resize", None)
    if resize_cfg is not None:
        resize = (int(resize_cfg[0]), int(resize_cfg[1]))
    else:
        resize = None

    img_t, (orig_H, orig_W), (proc_H, proc_W), orig_pil = preprocess_image(args.image, resize)
    img_t = img_t.to(device)

    # 7) Forward pass
    with torch.no_grad():
        c1, c2, c3, c4 = backbone(img_t)
        logits, _     = aspp_head([c1, c2, c3, c4])
        # logits: [1, total_classes, H/4, W/4]

        # Upsample logits to original size
        logits_full = F.interpolate(
            logits, size=(orig_H, orig_W), mode="bilinear", align_corners=False
        )  # [1, total_classes, orig_H, orig_W]

        # Split into fine / coarse / super channels
        fine_logits   = logits_full[:, :n_fine, :, :]                        # [1, n_fine, H, W]
        coarse_logits = logits_full[:, n_fine : n_fine + n_coarse, :, :]      # [1, n_coarse, H, W]
        if has_super:
            super_logits = logits_full[:, n_fine + n_coarse :, :, :]          # [1, n_super, H, W]

        # Compute argmax
        fine_pred   = fine_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)    # [H, W]
        coarse_pred = coarse_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        if has_super:
            super_pred = super_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # 8) Save raw masks and colored overlays
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # a) Raw masks (grayscale class indices)
    fine_out_path   = os.path.join(args.output_dir, f"{base_name}_fine.png")
    save_mask(fine_pred, fine_out_path)
    coarse_out_path = os.path.join(args.output_dir, f"{base_name}_coarse.png")
    save_mask(coarse_pred, coarse_out_path)
    if has_super:
        super_out_path = os.path.join(args.output_dir, f"{base_name}_super.png")
        save_mask(super_pred, super_out_path)

    print(f"→ Saved fine‐level mask to {fine_out_path}")
    print(f"→ Saved coarse‐level mask to {coarse_out_path}")
    if has_super:
        print(f"→ Saved super‐level mask to {super_out_path}")

    # b) Solid‐color masks with class indices drawn at centroids
    cmap_fine    = create_colormap(n_fine)
    color_fine   = mask_to_color_image(fine_pred, cmap_fine)
    color_fine   = draw_class_indices(fine_pred, color_fine)
    fine_color_path = os.path.join(args.output_dir, f"{base_name}_fine_color.png")
    color_fine.save(fine_color_path)
    print(f"→ Saved fine‐level color mask + indices to {fine_color_path}")

    cmap_coarse    = create_colormap(n_coarse)
    color_coarse   = mask_to_color_image(coarse_pred, cmap_coarse)
    color_coarse   = draw_class_indices(coarse_pred, color_coarse)
    coarse_color_path = os.path.join(args.output_dir, f"{base_name}_coarse_color.png")
    color_coarse.save(coarse_color_path)
    print(f"→ Saved coarse‐level color mask + indices to {coarse_color_path}")

    if has_super:
        cmap_super    = create_colormap(n_super)
        color_super   = mask_to_color_image(super_pred, cmap_super)
        color_super   = draw_class_indices(super_pred, color_super)
        super_color_path = os.path.join(args.output_dir, f"{base_name}_super_color.png")
        color_super.save(super_color_path)
        print(f"→ Saved super‐level color mask + indices to {super_color_path}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
