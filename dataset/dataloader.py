import os
import glob
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


def build_fine_to_level_map(map_cfg: list, n_fine: int) -> torch.Tensor:
    """
    Given a list of “ranges” like [[0,1], [2,4], [5,10], …],
    produce a LongTensor of length n_fine mapping each fine‐index → its level index.
    Each sublist must be either [lbl] or [start,end] (inclusive).
    """
    mapping = [-1] * n_fine
    for lvl, sub in enumerate(map_cfg):
        if len(sub) == 1:
            lbl = int(sub[0])
            assert 0 <= lbl < n_fine, f"Label {lbl} outside [0..{n_fine-1}]"
            mapping[lbl] = lvl
        elif len(sub) == 2:
            start, end = int(sub[0]), int(sub[1])
            assert 0 <= start <= end < n_fine, f"Range [{start},{end}] invalid"
            for i in range(start, end + 1):
                mapping[i] = lvl
        else:
            raise ValueError(f"Each entry must be [lbl] or [start,end], got {sub}")
    missing = [i for i, m in enumerate(mapping) if m < 0]
    if missing:
        raise ValueError(f"Fine‐labels not mapped: {missing}")
    return torch.tensor(mapping, dtype=torch.long)


class JointTransform:
    """
    Joint transform: resize + random horizontal flip + normalize.
    """
    def __init__(self, resize=None, hflip_prob=0.5,
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225)):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, img: Image.Image, mask: torch.Tensor):
        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=self.resize,
                mode="nearest"
            ).long().squeeze(0).squeeze(0)
        if torch.rand(1).item() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = torch.flip(mask, dims=[1])
        img_t = transforms.ToTensor()(img)
        img_t = transforms.Normalize(mean=self.normalize_mean,
                                     std=self.normalize_std)(img_t)
        return img_t, mask


class HieroDataloader(Dataset):
    """
    A hierarchical segmentation dataset that takes a `split` parameter in __init__.

    Args:
        config (str): Path to a YAML file defining dataset and class hierarchy.
        split (str): Either 'train' or 'val'. Controls which subfolder to load.
        transform (callable, optional): If provided, should take (PIL image, fine_mask_tensor)
                                        and return (image_tensor, fine_mask_tensor_resized).
                                        Otherwise, defaults to ToTensor+Normalize.

    __getitem__ returns:
      - If super_coarse_map is present: (img_tensor, fine_mask, coarse_mask, super_mask)
      - Otherwise:                    (img_tensor, fine_mask, coarse_mask)
    """
    def __init__(self, config, split='train', transform=None):
        # 1) Load YAML
        if isinstance(config, str):
            assert config.endswith(('.yaml', '.yml')), "Config must be a YAML file"
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            raise ValueError("A YAML config file path must be provided")
        self.cfg = cfg

        # 2) Validate split
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        self.split = split

        # 3) Parse dataset paths
        ds_cfg = cfg['dataset']
        root_dir = ds_cfg['root']

        # Strip leading slash so os.path.join works
        sub_img = ds_cfg[split]['image_subdir'].lstrip("/\\")
        sub_msk = ds_cfg[split]['mask_subdir'].lstrip("/\\")
        self.img_dir = os.path.join(root_dir, sub_img)
        self.msk_dir = os.path.join(root_dir, sub_msk)

        # 4) List filenames in each directory and intersect
        img_files = set(os.listdir(self.img_dir))
        msk_files = set(os.listdir(self.msk_dir))
        common_files = sorted(img_files & msk_files)
        assert common_files, f"No matching files in {self.img_dir} and {self.msk_dir}"

        # Build full paths in sorted order
        self.img_paths = [os.path.join(self.img_dir, fn) for fn in common_files]
        self.msk_paths = [os.path.join(self.msk_dir, fn) for fn in common_files]

        assert len(self.img_paths) == len(self.msk_paths), \
            f"Found {len(self.img_paths)} {split} images but {len(self.msk_paths)} {split} masks"

        # 5) Parse class hierarchy (fine → coarse → optional super)
        cls_cfg = cfg['classes']
        self.fine_names_dict = cls_cfg['fine_names']
        self.n_fine = len(self.fine_names_dict)

        self.coarse_map_cfg = cls_cfg['coarse_to_fine_map']
        self.fine_to_coarse = build_fine_to_level_map(self.coarse_map_cfg, self.n_fine)
        self.n_coarse = int(self.fine_to_coarse.max().item()) + 1
        self.coarse_names_dict = cls_cfg['coarse_names']

        if 'super_coarse_map' in cls_cfg:
            self.super_hiera = True
            self.super_map_cfg = cls_cfg['super_coarse_to_coarse_map']
            self.fine_to_super = build_fine_to_level_map(self.super_map_cfg, self.n_fine)
            self.n_super = int(self.fine_to_super.max().item()) + 1
            self.super_names_dict = cls_cfg['super_coarse_names']
        else:
            self.super_hiera = False
            self.fine_to_super = None
            self.n_super = None
            self.super_names_dict = None

        # 6) Set up transform
        if transform is not None:
            self.transform = transform
        else:
            tf_cfg = cfg.get('transform', {})
            if 'resize' in tf_cfg or 'hflip_prob' in tf_cfg:
                resize = None
                if tf_cfg.get('resize') is not None:
                    resize = (int(tf_cfg['resize'][0]), int(tf_cfg['resize'][1]))
                hflip_prob = float(tf_cfg.get('hflip_prob', 0.5))
                self.transform = JointTransform(resize=resize, hflip_prob=hflip_prob)
            else:
                self.transform = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1) Load image + fine‐mask
        img_path = self.img_paths[idx]
        msk_path = self.msk_paths[idx]
        img = Image.open(img_path).convert("RGB")
        fine_np = np.array(Image.open(msk_path), dtype=np.int64)  # shape (H, W)
        fine_mask = torch.from_numpy(fine_np).long()

        # 2) Compute coarse & super BEFORE any transform
        coarse_mask = self.fine_to_coarse[fine_mask]
        if self.super_hiera:
            super_mask = self.fine_to_super[fine_mask]
        else:
            super_mask = None

        # 3) Apply joint transform if provided
        if self.transform is not None:
            img_t, fine_mask = self.transform(img, fine_mask)
            coarse_mask = self.fine_to_coarse[fine_mask]
            if self.super_hiera:
                super_mask = self.fine_to_super[fine_mask]
        else:
            img_t = transforms.ToTensor()(img)
            img_t = transforms.Normalize((0.485,0.456,0.406),
                                         (0.229,0.224,0.225))(img_t)

        # 4) Return tuple
        if self.super_hiera:
            return img_t, fine_mask, coarse_mask, super_mask
        else:
            return img_t, fine_mask, coarse_mask
