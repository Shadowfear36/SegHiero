# train.py

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from terminaltables import AsciiTable

from dataset.dataloader import HieroDataloader

# Backbone imports
from models.backbone.resnet import ResNetBackbone
from models.backbone.convnext import ConvNeXtBackbone
from models.backbone.segformer import SegFormerBackbone

# Head imports
from models.head.sep_aspp_contrast_head import DepthwiseSeparableASPPContrastHead
from models.head.segformer_head import SegFormerHead, UltraFastSegFormerHead, ExtremelyFastSegFormerHead

# Two versions of the loss, one for 2-level and one for 3-level
from models.loss.hiera_triplet_loss import HieraTripletLoss
from models.loss.rmi_hiera_triplet_loss import RMIHieraTripletLoss


# --- FACTORY FUNCTIONS ---
def create_backbone(config):
    """Factory function to create backbone based on config"""
    backbone_config = config.get('backbone', {'type': 'resnet', 'depth': 101})
    backbone_type = backbone_config.get('type', 'resnet').lower()

    print(f"→ Creating {backbone_type} backbone...")

    if backbone_type == 'resnet':
        depth = backbone_config.get('depth', 101)
        pretrained = backbone_config.get('pretrained', True)
        return ResNetBackbone(depth=depth, pretrained=pretrained)
    elif backbone_type == 'convnext':
        variant = backbone_config.get('variant', 'tiny')
        pretrained = backbone_config.get('pretrained', True)
        return ConvNeXtBackbone(variant=variant, pretrained=pretrained)
    elif backbone_type == 'segformer':
        variant = backbone_config.get('variant', 'mit-b0')
        pretrained = backbone_config.get('pretrained', True)
        return SegFormerBackbone(variant=variant, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def get_backbone_channels(backbone_config):
    """Get channel dimensions for each backbone type"""
    backbone_type = backbone_config.get('type', 'resnet').lower()

    if backbone_type == 'resnet':
        depth = backbone_config.get('depth', 101)
        if depth in [50, 101, 152]:
            return {'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}
        elif depth in [18, 34]:
            return {'c1': 64, 'c2': 128, 'c3': 256, 'c4': 512}

    elif backbone_type == 'convnext':
        variant = backbone_config.get('variant', 'tiny')
        channels_map = {
            'tiny': {'c1': 96, 'c2': 192, 'c3': 384, 'c4': 768},
            'small': {'c1': 96, 'c2': 192, 'c3': 384, 'c4': 768},
            'base': {'c1': 128, 'c2': 256, 'c3': 512, 'c4': 1024},
            'large': {'c1': 192, 'c2': 384, 'c3': 768, 'c4': 1536},
            'xlarge': {'c1': 256, 'c2': 512, 'c3': 1024, 'c4': 2048}
        }
        return channels_map.get(variant, channels_map['tiny'])

    elif backbone_type == 'segformer':
        variant = backbone_config.get('variant', 'mit-b0')
        channels_map = {
            'mit-b0': {'c1': 32, 'c2': 64, 'c3': 160, 'c4': 256},
            'mit-b1': {'c1': 64, 'c2': 128, 'c3': 320, 'c4': 512},
            'mit-b2': {'c1': 64, 'c2': 128, 'c3': 320, 'c4': 512},
            'mit-b3': {'c1': 64, 'c2': 128, 'c3': 320, 'c4': 512},
            'mit-b4': {'c1': 64, 'c2': 128, 'c3': 320, 'c4': 512},
            'mit-b5': {'c1': 64, 'c2': 128, 'c3': 320, 'c4': 512}
        }
        return channels_map.get(variant, channels_map['mit-b0'])

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def create_head(config, in_channels_dict, num_classes):
    """Factory function to create segmentation head based on config"""
    head_config = config.get('head', {'type': 'aspp'})
    head_type = head_config.get('type', 'aspp').lower()

    print(f"→ Creating {head_type} head...")

    if head_type == 'aspp':
        # Get ASPP-specific parameters from model config
        model_cfg = config.get('model', {})
        return DepthwiseSeparableASPPContrastHead(
            in_channels=in_channels_dict['c4'],
            c1_in_channels=in_channels_dict['c1'],
            c1_channels=model_cfg.get('c1_channels', 48),
            aspp_channels=model_cfg.get('aspp_channels', 512),
            dilations=tuple(model_cfg.get('aspp_dilations', [1, 12, 24, 36])),
            num_classes=num_classes,
            proj_dim=model_cfg.get('projection_dim', 256),
            proj_type=model_cfg.get('projection_type', 'convmlp'),
        )

    elif head_type == 'segformer':
        in_channels_list = [in_channels_dict['c1'], in_channels_dict['c2'],
                           in_channels_dict['c3'], in_channels_dict['c4']]
        return SegFormerHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            proj_dim=head_config.get('proj_dim', 256)
        )

    elif head_type == 'ultrafast_segformer':
        in_channels_list = [in_channels_dict['c1'], in_channels_dict['c2'],
                           in_channels_dict['c3'], in_channels_dict['c4']]
        return UltraFastSegFormerHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            proj_dim=head_config.get('proj_dim', 128)
        )

    elif head_type == 'extremely_fast_segformer':
        in_channels_list = [in_channels_dict['c1'], in_channels_dict['c2'],
                           in_channels_dict['c3'], in_channels_dict['c4']]
        return ExtremelyFastSegFormerHead(
            in_channels_list=in_channels_list,
            num_classes=num_classes,
            proj_dim=head_config.get('proj_dim', 64)
        )

    else:
        available_heads = ['aspp', 'segformer', 'ultrafast_segformer', 'extremely_fast_segformer']
        raise ValueError(f"Unknown head type: {head_type}. Supported: {available_heads}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Hiera-Segmentation model using a single YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint (.pth) to initialize the model (optional)",
    )
    return parser.parse_args()


def compute_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index=255):
    """
    Compute pixel accuracy over non-ignored pixels.
    preds:   [B, H, W] (predicted class indices)
    targets: [B, H, W] (ground-truth class indices, 0..n_fine−1 or 255 to ignore)
    """
    valid = (targets != ignore_index)
    correct = (preds == targets) & valid
    total_correct = correct.sum().item()
    total_valid = valid.sum().item()
    if total_valid == 0:
        return 0.0
    return total_correct / total_valid


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


def main():
    args = parse_args()

    # 1) Load config.yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    
    # 1.1) If user supplied a “gpus” list, set CUDA_VISIBLE_DEVICES accordingly
    if "gpus" in cfg["training"]:
        gpu_list = cfg["training"]["gpus"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_list)
        print(f"→ Using GPUs {gpu_list}, primary device = {torch.device(cfg['training']['device'])}")

    device = torch.device(cfg["training"]["device"])
    # Immediately verify how many GPUs PyTorch sees
    num_visible = torch.cuda.device_count()
    print(f"[DEBUG] torch.cuda.device_count() = {num_visible}, torch.cuda.current_device() = {torch.cuda.current_device()}")

    # 2) Build Datasets + DataLoaders
    train_ds = HieroDataloader(args.config, split="train")
    val_ds   = HieroDataloader(args.config, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    print(f"Number of train samples: {len(train_ds)}")
    print(f"Number of val   samples: {len(val_ds)}")

    # 3) Count classes
    n_fine   = len(cfg["classes"]["fine_names"])
    n_coarse = len(cfg["classes"]["coarse_names"])
    has_super = "super_coarse_names" in cfg["classes"]
    n_super  = len(cfg["classes"].get("super_coarse_names", {}))

    # Depending on whether super_coarse is in the YAML, choose 2-level or 3-level loss
    if has_super:
        total_classes = n_fine + n_coarse + n_super
    else:
        total_classes = n_fine + n_coarse

    print(f"n_fine={n_fine}, n_coarse={n_coarse}, has_super={has_super}, n_super={n_super}")
    print(f"Total classes (output dim) = {total_classes}")

    # 4) Build Backbone + Head + (Optional) Aux Head using factory functions
    print("\n" + "="*60)
    print("BUILDING MODEL ARCHITECTURE")
    print("="*60)

    # Create backbone using factory
    backbone = create_backbone(cfg).to(device)
    backbone_channels = get_backbone_channels(cfg.get('backbone', {}))

    print(f"   Backbone channels: C1={backbone_channels['c1']}, C2={backbone_channels['c2']}, "
          f"C3={backbone_channels['c3']}, C4={backbone_channels['c4']}")

    # Create segmentation head using factory
    seg_head = create_head(cfg, backbone_channels, total_classes).to(device)

    # Auxiliary head on C3 for fine-class supervision only
    aux_head = nn.Sequential(
        nn.Conv2d(backbone_channels['c3'], n_fine, kernel_size=1, bias=False),
        nn.BatchNorm2d(n_fine),
        nn.ReLU(inplace=True),
    ).to(device)

    print(f"   Auxiliary head: C3({backbone_channels['c3']}) → {n_fine} classes")
    print("="*60 + "\n")

    # If multiple GPUs are visible, wrap each sub-module in DataParallel
    if num_visible > 1:
        print(f"→ Wrapping models in DataParallel (using {num_visible} GPUs)")
        backbone  = nn.DataParallel(backbone)
        seg_head  = nn.DataParallel(seg_head)
        aux_head  = nn.DataParallel(aux_head)

    # 4.1) If --pretrained was provided, load those weights now
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found at {args.pretrained}")
        print(f"→ Loading pretrained weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)

        # If DataParallel, state_dict keys are prefixed by "module.", so strip if necessary
        def _strip_module(prefix_dict):
            stripped = {}
            for k, v in prefix_dict.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                stripped[new_k] = v
            return stripped

        backbone_state = _strip_module(ckpt["backbone_state_dict"])

        # Try different possible keys for the segmentation head
        seg_head_keys = ['seg_head_state_dict', 'seg_head_state_dict', 'main_head_state_dict']
        seg_head_loaded = False

        for key in seg_head_keys:
            if key in ckpt:
                try:
                    seg_head_state = _strip_module(ckpt[key])
                    seg_head.load_state_dict(seg_head_state)
                    print(f"→ Segmentation head weights loaded from '{key}'")
                    seg_head_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Could not load seg head from '{key}': {e}")

        if not seg_head_loaded:
            print("⚠️  WARNING: Could not load segmentation head weights from checkpoint!")

        # Load auxiliary head
        if 'aux_head_state_dict' in ckpt:
            aux_state = _strip_module(ckpt["aux_head_state_dict"])
            aux_head.load_state_dict(aux_state)
            print("→ Auxiliary head weights loaded")
        else:
            print("⚠️  WARNING: No auxiliary head weights in checkpoint!")

        backbone.load_state_dict(backbone_state)
        print("→ Pretrained weights successfully loaded (backbone + heads)")

    # 5) Build Loss function (either 2-level or 3-level)
    if not has_super:
        # ────────────────────────────────────────────────────────
        # 2-level case:
        #   - build “fine→coarse” mapping (as a Python list) from YAML
        #   - build “hiera_index” list so that HieraTripletLoss can consume it
        # ────────────────────────────────────────────────────────
        coarse_to_fine_cfg = cfg["classes"]["coarse_to_fine_map"]

        # Build a torch.LongTensor of length=n_fine mapping each fine→coarse
        fine_to_coarse = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine)

        # Build hiera_index = [[start,end+1],...] exactly as HieraTripletLoss expects
        hiera_index = build_hiera_index(coarse_to_fine_cfg)

        # Convert “fine_to_coarse” to a plain Python list for HieraTripletLoss
        hiera_map_list = fine_to_coarse.tolist()

        # Get triplet config parameters
        triplet_cfg = train_cfg.get("triplet", {})

        hiera_loss_fn = HieraTripletLoss(
            num_classes=n_fine,
            hiera_map=hiera_map_list,
            hiera_index=hiera_index,
            ignore_index=255,
            use_sigmoid=False,
            loss_weight=train_cfg.get("fine_weight", 1.0),
            hiera_loss_multiplier=train_cfg.get("hiera_loss_multiplier", 5.0),
            triplet_margin=triplet_cfg.get("margin", 0.6),
            triplet_max_samples=triplet_cfg.get("max_samples", 200),
            triplet_warmup_steps=triplet_cfg.get("schedule_warmup_steps", 80000),
            triplet_min_factor=triplet_cfg.get("schedule_min_factor", 0.25),
            triplet_max_factor=triplet_cfg.get("schedule_max_factor", 0.5),
        ).to(device)

        if num_visible > 1:
            hiera_loss_fn = nn.DataParallel(hiera_loss_fn)

    else:
        # ────────────────────────────────────────────────────────
        # 3-level case:
        #   - build fine→mid (coarse) and fine→high (super) maps
        #   - supply n_fine, n_mid, n_high, plus the two mapping-tensors
        # ────────────────────────────────────────────────────────
        coarse_to_fine_cfg  = cfg["classes"]["coarse_to_fine_map"]
        super_to_coarse_cfg = cfg["classes"]["super_coarse_to_coarse_map"]

        # 1) fine→mid
        fine_to_mid = build_fine_to_coarse_map(coarse_to_fine_cfg, n_fine)

        # 2) fine→high
        fine_to_high = build_fine_to_super_map(super_to_coarse_cfg, n_fine)

        n_mid  = n_coarse
        n_high = n_super

        # Get triplet config parameters
        triplet_cfg = train_cfg.get("triplet", {})
        triplet_3level_cfg = triplet_cfg.get("schedule_warmup_steps_3level", {})

        hiera_loss_fn = RMIHieraTripletLoss(
            n_fine            = n_fine,
            n_mid             = n_mid,
            n_high            = n_high,
            fine_to_mid       = fine_to_mid,
            fine_to_high      = fine_to_high,
            rmi_radius        = train_cfg.get("rmi_radius", 3),
            rmi_pool_way      = train_cfg.get("rmi_pool_way", 0),
            rmi_pool_size     = train_cfg.get("rmi_pool_size", 3),
            rmi_pool_stride   = train_cfg.get("rmi_pool_stride", 3),
            loss_weight_lambda= train_cfg.get("rmi_loss_weight_lambda", 0.5),
            loss_weight       = 1.0,
            ignore_index      = 255,
            hiera_loss_multiplier=train_cfg.get("hiera_loss_multiplier", 5.0),
            triplet_margin=triplet_cfg.get("margin", 0.6),
            triplet_max_samples=triplet_cfg.get("max_samples", 200),
            triplet_warmup_steps_small=triplet_3level_cfg.get("small", 60000),
            triplet_warmup_steps_large=triplet_3level_cfg.get("large", 160000),
            triplet_min_factor=triplet_cfg.get("schedule_min_factor", 0.25),
            triplet_max_factor=triplet_cfg.get("schedule_max_factor", 0.5),
        ).to(device)

        if num_visible > 1:
            hiera_loss_fn = nn.DataParallel(hiera_loss_fn)

    # 6) Auxiliary-only criterion for C3 → fine-classes
    aux_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 7) Optimizer
    optimizer = optim.SGD(
        list(backbone.parameters())
        + list(seg_head.parameters())
        + list(aux_head.parameters()),
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    best_val_loss = float("inf")
    history = []

    # 8) Training loop
    for epoch in range(cfg["training"]["epochs"]):
        backbone.train()
        seg_head.train()
        aux_head.train()

        running_train_loss = 0.0
        train_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]")

        for batch in pbar:
            img_t      = batch[0].to(device, non_blocking=True)  # [B, 3, H, W]
            fine_mask  = batch[1].to(device, non_blocking=True)  # [B, H, W]

            optimizer.zero_grad()

            # 1) Forward through backbone
            c1, c2, c3, c4 = backbone(img_t)

            # 2) Main head → logits + embedding
            main_logits, embedding = seg_head([c1, c2, c3, c4])
            # main_logits: [B, total_classes, H/4, W/4]

            B, _, H4, W4 = main_logits.shape
            H, W = fine_mask.shape[-2:]

            # Build logit_before (downsampled by 2 → 1/8)
            logit_before_full = nn.functional.interpolate(
                main_logits, scale_factor=0.5, mode="bilinear", align_corners=False
            )  # [B, total_classes, H/8, W/8]

            # Build logit_after (upsampled to full resolution H × W)
            logit_after_full = nn.functional.interpolate(
                main_logits, size=(H, W), mode="bilinear", align_corners=False
            )  # [B, total_classes, H, W]

            # 3) Compute hierarchical loss
            step_tensor = torch.tensor([epoch], dtype=torch.long, device=device)

            if not has_super:
                # 2-level: pass full logits for fine+coarse to the loss
                main_loss = hiera_loss_fn(
                    step_tensor,
                    embedding,                              # [B, D, H/8, W/8]
                    logit_before_full[:, :n_fine, :, :],    # just the fine slice before (for triplet)
                    logit_after_full,                       # full [B, n_fine+n_coarse, H, W]
                    fine_mask
                )
            else:
                # 3-level: pass full logits for fine+mid+high
                main_loss = hiera_loss_fn(
                    step_tensor,
                    embedding,
                    logit_before_full[:, :n_fine, :, :],    # fine slice before (triplet only)
                    logit_after_full,                       # full [B, n_fine+n_mid+n_high, H, W]
                    fine_mask
                )

            # 4) Auxiliary head on c3 (fine classes only)
            aux_logits = aux_head(c3)  # [B, n_fine, H/16, W/16]
            aux_logits = nn.functional.interpolate(
                aux_logits, size=(H, W), mode="bilinear", align_corners=False
            )  # [B, n_fine, H, W]
            aux_loss = aux_criterion(aux_logits, fine_mask)

            aux_loss_weight = train_cfg.get("aux_loss_weight", 0.4)
            loss = main_loss + aux_loss_weight * aux_loss
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix(train_loss=running_train_loss / (pbar.n + 1))

        avg_train_loss = running_train_loss / train_batches

        # -------------------------------
        # 9) Validation
        # -------------------------------
        backbone.eval()
        seg_head.eval()
        aux_head.eval()

        running_val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Val]  ")
            for batch in pbar_val:
                img_t     = batch[0].to(device, non_blocking=True)
                fine_mask = batch[1].to(device, non_blocking=True)

                c1, c2, c3, c4 = backbone(img_t)
                main_logits, embedding = seg_head([c1, c2, c3, c4])

                B, _, H4, W4 = main_logits.shape
                H, W = fine_mask.shape[-2:]

                logit_before_full = nn.functional.interpolate(
                    main_logits, scale_factor=0.5, mode="bilinear", align_corners=False
                )
                logit_after_full = nn.functional.interpolate(
                    main_logits, size=(H, W), mode="bilinear", align_corners=False
                )

                step_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
                if not has_super:
                    main_loss = hiera_loss_fn(
                    step_tensor,
                    embedding,
                    logit_before_full[:, :n_fine, :, :],
                    logit_after_full,
                    fine_mask
                )
                else:
                    main_loss = hiera_loss_fn(
                        step_tensor,
                        embedding,
                        logit_before_full[:, :n_fine, :, :],
                        logit_after_full,
                        fine_mask
                    )

                aux_logits = aux_head(c3)
                aux_logits = nn.functional.interpolate(
                    aux_logits, size=(H, W), mode="bilinear", align_corners=False
                )
                aux_loss = aux_criterion(aux_logits, fine_mask)

                aux_loss_weight = train_cfg.get("aux_loss_weight", 0.4)
                loss = main_loss + aux_loss_weight * aux_loss
                running_val_loss += loss.item()

                # Compute pixel-accuracy on fine level
                fine_pred = logit_after_full[:, :n_fine, :, :].argmax(dim=1)
                batch_correct = (fine_pred == fine_mask) & (fine_mask != 255)
                correct_pixels += batch_correct.sum().item()
                total_pixels += (fine_mask != 255).sum().item()

                pbar_val.set_postfix(
                    val_loss=running_val_loss / (pbar_val.n + 1),
                    val_acc=correct_pixels / max(total_pixels, 1)
                )

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct_pixels / max(total_pixels, 1)

        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

        # Build a simple table of the latest epoch metrics
        table_data = [
            ["Epoch", "Avg Train Loss", "Avg Val Loss", "Val Pixel Acc"]
        ]
        last = history[-1]
        table_data.append([
            str(last["epoch"]),
            f"{last['train_loss']:.4f}",
            f"{last['val_loss']:.4f}",
            f"{last['val_acc'] * 100:.2f}%"
        ])
        print(AsciiTable(table_data).table)

        # -------------------------------
        # 10) Checkpoint
        # -------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                "epoch": last["epoch"],
                "backbone_state_dict": backbone.module.state_dict() if num_visible > 1 else backbone.state_dict(),
                "seg_head_state_dict": seg_head.module.state_dict() if num_visible > 1 else seg_head.state_dict(),
                "aux_head_state_dict": aux_head.module.state_dict() if num_visible > 1 else aux_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "backbone_config": cfg.get('backbone', {}),  # Save backbone config for inference
                "head_config": cfg.get('head', {})  # Save head config for inference
            }
            os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
            save_path = os.path.join(
                cfg["output"]["checkpoint_dir"],
                f"{cfg['output']['project_name']}epoch_{epoch}_best.pth"
            )
            torch.save(ckpt, save_path)
            print(f"→ Saved new best model to {save_path}\n")

    print("Training complete.")

    # =============================================================================
    # 11) Final per-class accuracy on validation set (fine-level classes only)
    # =============================================================================
    print("\n## Computing per-fine-class pixel accuracy on val set ...\n")
    backbone.eval()
    seg_head.eval()

    per_class_correct = torch.zeros(n_fine, dtype=torch.long, device=device)
    per_class_total   = torch.zeros(n_fine, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final eval on val set"):
            img_t     = batch[0].to(device, non_blocking=True)
            fine_mask = batch[1].to(device, non_blocking=True)  # [B, H, W]

            c1, c2, c3, c4 = backbone(img_t)
            logits, _ = seg_head([c1, c2, c3, c4])

            B, _, H4, W4 = logits.shape
            H, W = fine_mask.shape[-2:]

            logit_full = nn.functional.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )  # [B, total_classes, H, W]

            fine_logits = logit_full[:, :n_fine, :, :]  # [B, n_fine, H, W]
            fine_pred   = fine_logits.argmax(dim=1)     # [B, H, W]

            for i in range(n_fine):
                mask_i = (fine_mask == i)
                per_class_total[i] += mask_i.sum().item()
                per_class_correct[i] += ((fine_pred == i) & mask_i).sum().item()

    table_data = [["Class ID", "Class Name", "Pixel Acc (%)"]]
    fine_names = cfg["classes"]["fine_names"]
    for i in range(n_fine):
        total_i   = per_class_total[i].item()
        correct_i = per_class_correct[i].item()
        acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0.0
        class_name = fine_names[i]
        table_data.append([str(i), class_name, f"{acc_i:.2f}"])

    print(AsciiTable(table_data).table)


if __name__ == "__main__":
    main()
