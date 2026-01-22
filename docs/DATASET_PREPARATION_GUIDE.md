# Dataset Preparation Guide for SegHiero

This guide walks through how to prepare a dataset for hierarchical semantic segmentation with SegHiero.

## Overview

SegHiero needs:
1. RGB images
2. Fine-level pixel masks (0 to n_fine-1, with 255 for ignore)
3. A hierarchical class structure defined in your config file

The key difference from standard segmentation: you define how fine classes group into coarse (and optionally super-coarse) categories through config, not in the masks themselves.

## Quick Start

**Minimum requirements:**
```
your_dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── val/
    ├── images/
    │   └── ...
    └── masks/
        └── ...
```

**Key points:**
- Image and mask filenames must match exactly (same name, different extensions OK)
- Masks are single-channel grayscale PNG files
- Each pixel value represents a fine-level class (0 to n_fine-1)
- Use 255 for pixels you want to ignore during training

## Step-by-Step Dataset Preparation

### 1. Define Your Class Hierarchy

Before creating any masks, plan your hierarchy on paper.

**Example: Defect Detection**
```
Fine Level (8 classes):
- 0: Background
- 1: Crack_Small
- 2: Crack_Large
- 3: Rust_Surface
- 4: Rust_Deep
- 5: Dent_Minor
- 6: Dent_Major
- 7: Scratch

Coarse Level (4 groups):
- Group 0: Background → [0]
- Group 1: Cracks → [1, 2]
- Group 2: Rust → [3, 4]
- Group 3: Dents → [5, 6]
- (Note: Scratch doesn't fit well, so it gets its own group or maps to background)
```

**Example: Natural Scenes**
```
Fine Level (12 classes):
- 0: Sky
- 1: Road_Asphalt
- 2: Road_Concrete
- 3: Sidewalk
- 4: Car
- 5: Truck
- 6: Bus
- 7: Person
- 8: Cyclist
- 9: Tree
- 10: Building
- 11: Sign

Coarse Level (6 groups):
- Group 0: Sky → [0]
- Group 1: Road → [1, 2, 3]
- Group 2: Vehicles → [4, 5, 6]
- Group 3: People → [7, 8]
- Group 4: Nature → [9]
- Group 5: Infrastructure → [10, 11]
```

### 2. Create Fine-Level Masks

You only need to label at the fine level. The coarse groupings are defined in the config file.

**Tools for annotation:**
- [CVAT](https://cvat.org/) - Web-based, supports polygon/brush annotation
- [Labelme](https://github.com/wkentaro/labelme) - Simple desktop tool
- [LabelImg](https://github.com/tzutalin/labelImg) - For bounding boxes
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - For auto-segmentation
- Custom scripts (see examples below)

**Mask requirements:**
- Format: Single-channel grayscale PNG (8-bit or 16-bit)
- Pixel values: 0 to (n_fine - 1)
- Ignore value: 255 for uncertain/unlabeled regions
- Same dimensions as corresponding image (or will be resized during training)

**Creating masks in Python:**
```python
import numpy as np
from PIL import Image

# Create a blank mask (same size as your image)
mask = np.zeros((height, width), dtype=np.uint8)

# Label regions with fine-level class IDs
mask[100:200, 100:200] = 1  # Region belongs to class 1
mask[300:400, 300:400] = 2  # Region belongs to class 2

# Mark uncertain regions as ignore
mask[500:550, 500:550] = 255

# Save as PNG
Image.fromarray(mask).save('mask_001.png')
```

**Verifying your masks:**
```python
import numpy as np
from PIL import Image

mask = np.array(Image.open('mask_001.png'))
unique_values = np.unique(mask)
print(f"Unique values in mask: {unique_values}")
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")

# Should see values from 0 to n_fine-1, possibly 255
# Should NOT see values > n_fine-1 (except 255)
```

### 3. Organize Your Dataset

**Standard structure:**
```
dataset_root/
├── train/
│   ├── images/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   ├── 00003.jpg
│   │   └── ...
│   └── masks/
│       ├── 00001.png
│       ├── 00002.png
│       ├── 00003.png
│       └── ...
└── val/
    ├── images/
    │   ├── 10001.jpg
    │   ├── 10002.jpg
    │   └── ...
    └── masks/
        ├── 10001.png
        ├── 10002.png
        └── ...
```

**Naming conventions:**
- Images and masks must have matching filenames (extensions can differ)
- Use consistent naming: `00001.jpg` + `00001.png` ✓
- Avoid spaces in filenames
- Case matters on Linux/Mac

**Train/val split:**
- Typical split: 80% train, 20% val
- For small datasets: 70% train, 30% val
- For large datasets: 90% train, 10% val
- Make sure both splits have all classes represented

**Script to verify dataset structure:**
```python
import os
from pathlib import Path

def verify_dataset(root_path):
    root = Path(root_path)

    for split in ['train', 'val']:
        img_dir = root / split / 'images'
        mask_dir = root / split / 'masks'

        if not img_dir.exists():
            print(f"❌ Missing: {img_dir}")
            continue
        if not mask_dir.exists():
            print(f"❌ Missing: {mask_dir}")
            continue

        images = set(f.stem for f in img_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg'])
        masks = set(f.stem for f in mask_dir.iterdir() if f.suffix == '.png')

        print(f"\n{split.upper()} split:")
        print(f"  Images: {len(images)}")
        print(f"  Masks: {len(masks)}")

        missing_masks = images - masks
        extra_masks = masks - images

        if missing_masks:
            print(f"  ⚠️  Images missing masks: {len(missing_masks)}")
            for name in list(missing_masks)[:5]:
                print(f"     - {name}")

        if extra_masks:
            print(f"  ⚠️  Masks without images: {len(extra_masks)}")
            for name in list(extra_masks)[:5]:
                print(f"     - {name}")

        if not missing_masks and not extra_masks:
            print(f"  ✓ All images have matching masks")

verify_dataset('/path/to/your/dataset')
```

### 4. Configure the Hierarchy in YAML

Now define how your fine classes group into coarse categories.

**Two-level hierarchy example:**
```yaml
classes:
  n_fine: 8
  n_coarse: 4

  # Define how fine classes group into coarse
  coarse_to_fine_map:
    - [0]        # Coarse 0 = Fine class 0 (Background)
    - [1, 2]     # Coarse 1 = Fine classes 1-2 (Cracks)
    - [3, 4]     # Coarse 2 = Fine classes 3-4 (Rust)
    - [5, 6, 7]  # Coarse 3 = Fine classes 5-7 (Dents + Scratch)

  # Optional: Give names to your classes
  fine_names:
    0: Background
    1: Crack_Small
    2: Crack_Large
    3: Rust_Surface
    4: Rust_Deep
    5: Dent_Minor
    6: Dent_Major
    7: Scratch

  coarse_names:
    0: Background
    1: Cracks
    2: Rust
    3: Deformation
```

**Three-level hierarchy example:**
```yaml
classes:
  n_fine: 12
  n_coarse: 6
  n_super: 3

  # Fine → Coarse mapping
  coarse_to_fine_map:
    - [0]           # Coarse 0 = Sky
    - [1, 2, 3]     # Coarse 1 = All roads
    - [4, 5, 6]     # Coarse 2 = Vehicles
    - [7, 8]        # Coarse 3 = People
    - [9]           # Coarse 4 = Nature
    - [10, 11]      # Coarse 5 = Infrastructure

  # Coarse → Super mapping
  super_coarse_to_coarse_map:
    - [0]           # Super 0 = Sky (Coarse 0)
    - [1, 4, 5]     # Super 1 = Outdoor (Roads, Nature, Infrastructure)
    - [2, 3]        # Super 2 = Entities (Vehicles, People)

  fine_names:
    0: Sky
    1: Road_Asphalt
    2: Road_Concrete
    # ... etc

  coarse_names:
    0: Sky
    1: Road
    2: Vehicles
    # ... etc

  super_coarse_names:
    0: Sky
    1: Outdoor
    2: Entities
```

### 5. Set Dataset Path in Config

```yaml
dataset:
  root: '/absolute/path/to/your_dataset'  # Use absolute paths
  train:
    image_folder: 'train/images'
    mask_folder: 'train/masks'
  val:
    image_folder: 'val/images'
    mask_folder: 'val/masks'
```

## Common Issues and Solutions

### Issue: "Mask values exceed n_fine"

**Problem:** Your masks contain values larger than expected.

**Check:**
```python
import numpy as np
from PIL import Image
import glob

max_value = 0
for mask_path in glob.glob('dataset/train/masks/*.png'):
    mask = np.array(Image.open(mask_path))
    mask_max = mask[mask != 255].max()  # Ignore the 255s
    max_value = max(max_value, mask_max)

print(f"Maximum mask value: {max_value}")
print(f"Your n_fine should be at least: {max_value + 1}")
```

**Solution:** Update `n_fine` in your config, or fix your masks if the values are wrong.

### Issue: "Images and masks don't match"

**Problem:** Filenames don't correspond or dimensions differ.

**Check dimensions:**
```python
from PIL import Image
import os

img_path = 'dataset/train/images/00001.jpg'
mask_path = 'dataset/train/masks/00001.png'

img = Image.open(img_path)
mask = Image.open(mask_path)

print(f"Image size: {img.size}")
print(f"Mask size: {mask.size}")
print(f"Image mode: {img.mode}")  # Should be RGB
print(f"Mask mode: {mask.mode}")   # Should be L (grayscale)
```

**Solution:** SegHiero will resize both to match during training, but it's better to fix dimensions beforehand if they're way off.

### Issue: "Not enough training data"

**Augmentation strategies:**
- Random horizontal flipping (built into SegHiero)
- Random crops (modify dataloader)
- Color jittering (modify dataloader)
- Use data from similar domains
- Synthetic data generation

**Minimum viable dataset:**
- For testing: 50-100 images
- For real training: 500+ images
- For good performance: 1000+ images

### Issue: "Class imbalance"

Some classes appear much more than others (e.g., background is 90% of pixels).

**Solutions:**
1. **Weighted loss** - Modify loss function to weight rare classes higher
2. **Sampling** - Over-sample images with rare classes
3. **Crop strategy** - Center crops on rare class regions
4. **More data** - Collect more examples of rare classes

## Dataset Statistics

Before training, check your dataset statistics:

```python
import numpy as np
from PIL import Image
import glob
from collections import Counter

def analyze_dataset(mask_dir):
    all_pixels = []

    for mask_path in glob.glob(f'{mask_dir}/*.png'):
        mask = np.array(Image.open(mask_path))
        pixels = mask[mask != 255].flatten()  # Exclude ignore pixels
        all_pixels.extend(pixels.tolist())

    counter = Counter(all_pixels)
    total = len(all_pixels)

    print(f"Total labeled pixels: {total:,}")
    print(f"\nClass distribution:")
    for class_id in sorted(counter.keys()):
        count = counter[class_id]
        pct = 100.0 * count / total
        print(f"  Class {class_id}: {count:,} pixels ({pct:.2f}%)")

analyze_dataset('dataset/train/masks')
```

This helps you understand if you have severe class imbalance.

## Example: Converting from COCO Format

SegHiero includes a conversion script for COCO datasets. See [tools/README.md](tools/README.md) for full documentation.

**Quick usage:**

```bash
# Step 1: Generate template mapping config
python tools/coco_to_seghiero.py \
    --coco-json annotations/instances_train2017.json \
    --create-template tools/my_coco_mapping.yaml

# Step 2: Edit my_coco_mapping.yaml to define your fine classes

# Step 3: Convert the dataset
python tools/coco_to_seghiero.py \
    --coco-json annotations/instances_train2017.json \
    --images-dir images/train2017 \
    --output-dir dataset/train/masks \
    --mapping-config tools/my_coco_mapping.yaml \
    --verify
```

**Or write your own converter:**

```python
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

def coco_to_seghiero(coco_json, image_dir, output_dir, category_mapping):
    """
    Convert COCO annotations to SegHiero format.

    category_mapping: dict mapping COCO category_id to fine class ID
    Example: {1: 0, 2: 1, 3: 1, 4: 2, ...}
    """
    coco = COCO(coco_json)

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        # Start with background (class 0) or ignore (255)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Draw each annotation
        for ann in anns:
            cat_id = ann['category_id']
            fine_class = category_mapping.get(cat_id, 255)  # 255 if unknown

            # Convert COCO segmentation to binary mask
            rle = coco.annToRLE(ann)
            binary_mask = coco_mask.decode(rle)

            # Apply to our mask
            mask[binary_mask == 1] = fine_class

        # Save mask
        output_path = f"{output_dir}/{img_info['file_name'].replace('.jpg', '.png')}"
        Image.fromarray(mask).save(output_path)

# Example usage
category_mapping = {
    1: 0,   # COCO person → Fine class 0
    2: 1,   # COCO bicycle → Fine class 1
    3: 2,   # COCO car → Fine class 2
    # ... etc
}

coco_to_seghiero(
    'annotations/instances_train2017.json',
    'images/train2017',
    'dataset/train/masks',
    category_mapping
)
```

## Example: Creating Masks from Labelme JSON

```python
import json
import numpy as np
from PIL import Image, ImageDraw

def labelme_to_mask(json_path, output_path, class_to_id):
    """
    Convert Labelme JSON to SegHiero mask.

    class_to_id: dict mapping label names to fine class IDs
    Example: {'background': 0, 'crack': 1, 'rust': 2}
    """
    with open(json_path) as f:
        data = json.load(f)

    h, w = data['imageHeight'], data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        class_id = class_to_id.get(label, 255)

        # Create polygon mask
        img_pil = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img_pil).polygon([tuple(p) for p in points],
                                        outline=class_id, fill=class_id)
        polygon_mask = np.array(img_pil)

        # Merge into main mask
        mask[polygon_mask == class_id] = class_id

    Image.fromarray(mask).save(output_path)

# Example usage
class_to_id = {
    'background': 0,
    'crack_small': 1,
    'crack_large': 2,
    'rust': 3,
}

labelme_to_mask('annotations/image_001.json',
                'dataset/train/masks/image_001.png',
                class_to_id)
```

## Checklist Before Training

- [ ] Dataset directory structure is correct
- [ ] All images have corresponding masks
- [ ] Mask values are in range [0, n_fine-1] plus 255 for ignore
- [ ] Masks are single-channel grayscale PNG
- [ ] Config file has correct `dataset.root` path
- [ ] Config file has correct `n_fine` value
- [ ] Hierarchy mapping (`coarse_to_fine_map`) covers all fine classes
- [ ] Train/val split is reasonable (typically 80/20)
- [ ] Checked class distribution (no extreme imbalance)
- [ ] Verified a few masks visually to ensure labels are correct

## Visualizing Your Dataset

Quick script to check if your masks look correct:

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_sample(img_path, mask_path, n_fine):
    img = Image.open(img_path)
    mask = np.array(Image.open(mask_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Mask (raw values)
    axes[1].imshow(mask, cmap='tab20', vmin=0, vmax=n_fine)
    axes[1].set_title(f'Mask (Classes 0-{n_fine-1})')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(mask, alpha=0.5, cmap='tab20', vmin=0, vmax=n_fine)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

visualize_sample(
    'dataset/train/images/00001.jpg',
    'dataset/train/masks/00001.png',
    n_fine=8
)
```

## Next Steps

Once your dataset is ready:

1. Update your config file with the correct paths and hierarchy
2. Run a quick test with 1 epoch to verify everything loads correctly
3. Check the output logs for any warnings about class distributions
4. Start full training!

```bash
python train.py --config your-config.yaml
```

If you run into issues, check the error messages carefully - they usually point to specific problems with your dataset structure or config file.
