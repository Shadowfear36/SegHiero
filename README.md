# SegHiero: Hierarchical Semantic Segmentation

SegHiero is a PyTorch‐based framework for hierarchical semantic segmentation. Instead of predicting a single “flat” set of classes, SegHiero allows you to define a multi‐level label hierarchy (e.g. fine→coarse or fine→mid→high) so that the model simultaneously learns to predict:

1. **Fine‐level labels** (e.g. every distinct object or defect type),  
2. **Coarse‐level labels** (grouping related fine‐labels),  
3. **Optional super‐coarse labels** (grouping coarse categories into higher‐level buckets).

This README explains:
- What hierarchical semantic segmentation is and why it helps
- How to configure a two‐level or three‐level hierarchy via YAML
- The structure of the project and how to train/evaluate a model
- How the config file is interpreted by `train.py`
- Tips for dataset preparation, custom backbones, and inference

---

## 🌟 New: Multi-Architecture Support

SegHiero now supports **multiple backbones and segmentation heads** that can be easily swapped via YAML configuration:

**Supported Backbones:**
- ResNet (18, 34, 50, 101, 152) - Proven stability
- ConvNeXt (tiny, small, base, large, xlarge) - Modern CNN performance
- SegFormer (MiT-B0 through MiT-B5) - State-of-the-art transformer

**Supported Heads:**
- ASPP Head - Multi-scale context (works with all backbones)
- SegFormer Head - Best accuracy with SegFormer backbone
- UltraFast SegFormer Head - ~40% faster, <2% accuracy loss
- Extremely Fast SegFormer Head - ~60% faster for real-time applications

See [ARCHITECTURE_INTEGRATION_GUIDE.md](ARCHITECTURE_INTEGRATION_GUIDE.md) for detailed information and [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) for copy-paste configurations.

---

## Table of Contents

1. [What Is Hierarchical Semantic Segmentation?](#what-is-hierarchical-semantic-segmentation)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset & Label Preparation](#dataset--label-preparation) - See also: [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)
5. [**Architecture Configuration (NEW)**](#architecture-configuration)
6. [Config File Format](#config-file-format)
   - [1. Dataset Section](#1-dataset-section)
   - [2. Classes Section](#2-classes-section)
   - [3. Backbone Section (NEW)](#3-backbone-section)
   - [4. Head Section (NEW)](#4-head-section)
   - [5. Model Section](#5-model-section)
   - [6. Training Section](#6-training-section)
   - [7. Transform Section](#7-transform-section)
   - [8. Output Section](#8-output-section)
7. [Training & Validation](#training--validation)
8. [Inference](#inference)
9. [Custom Backbones / Heads](#custom-backbones--heads)
10. [License & Acknowledgments](#license--acknowledgments)

---

## What Is Hierarchical Semantic Segmentation?

Standard semantic segmentation predicts one label per pixel from a “flat” set of classes ⁠— for example, `{ background, car, person, road, sky, … }`. In many applications (e.g. medical, industrial inspection, autonomous driving), classes naturally organize into coarse groups:

- **Fine level**: very granular classes (e.g. “Kernel”, “Split Kernel”, “Kernel Piece”, “Decay”, …)  
- **Coarse level**: higher‐level buckets grouping those fine labels (e.g. all kernel defects grouped under “Kernel‐Bad”)  
- **Super / Ultra‐coarse level** (optional): group multiple coarse categories (e.g. “Kernel” + “InShell” under “Nut” vs. “Foreign Material”)

A **hierarchical** approach has several benefits:

1. **Multi‐scale supervision**: the model learns to predict both fine and coarse labels simultaneously, which can regularize against noisy fine annotations.  
2. **Error‐tolerant inference**: if the fine prediction is uncertain, the coarse prediction may still be correct.  
3. **Faster downstream tasks**: if an application only needs coarse labels, you can read them directly without post‐processing.  

SegHiero implements two popular strategies:

- **Two‐level hierarchy (fine→coarse)**:  
  - Loss includes a fine‐level binary cross‐entropy term (for each fine class), plus a “hierarchical” BCE that enforces consistency between fine scores and coarse scores.  
  - Optionally a triplet‐embedding loss to encourage separation of defect groups at the feature level.

- **Three‐level hierarchy (fine→mid→high)**:  
  - Same idea extended to three levels: fine → mid (coarse) → high (super‐coarse).  
  - RMI (Regional Mutual Information) loss can be used for more robust spatial consistency.

---

## Project Structure
```
SegHiero/
├── dataset/
│   └── dataloader.py                  # HieroDataloader implementation
│
├── models/
│   ├── backbone/
│   │   ├── resnet.py                  # ResNet (18, 34, 50, 101, 152)
│   │   ├── convnext.py                # ConvNeXt (NEW)
│   │   └── segformer.py               # SegFormer MiT (NEW)
│   ├── head/
│   │   ├── sep_aspp_contrast_head.py  # ASPP Head
│   │   └── segformer_head.py          # SegFormer Heads (NEW - 3 variants)
│   └── loss/
│       ├── cross_entropy_loss.py      # CE Loss wrapper
│       ├── tree_triplet_loss.py       # Hierarchical triplet loss
│       ├── hiera_triplet_loss.py      # 2-level HieraTripletLoss
│       └── rmi_hiera_triplet_loss.py  # 3-level RMIHieraTripletLoss
│
├── train.py                           # Main training script (with factory functions)
├── infer.py                           # Inference script
├── example-config.yaml                # Example config with all architectures
├── requirements.txt                   # Dependencies
├── tools/
│   ├── coco_to_seghiero.py            # COCO converter script (NEW)
│   ├── coco_mapping_example.yaml      # Example COCO mapping (NEW)
│   └── README.md                      # Tools documentation (NEW)
├── ARCHITECTURE_INTEGRATION_GUIDE.md  # Comprehensive architecture guide (NEW)
├── CONFIG_EXAMPLES.md                 # Quick config examples (NEW)
├── DATASET_PREPARATION_GUIDE.md       # Dataset creation guide (NEW)
└── CONTRIBUTING.md                    # How to contribute
```



- **`dataset/dataloader.py`**  
  - Defines `HieroDataloader`, which loads images and “fine‐level” masks.  
  - Automatically builds coarse and (optionally) super‐coarse targets using precomputed mappings.  
  - Applies joint transforms (resize, random flip, normalize).

- **`models/backbone/resnet.py`**  
  - A PyTorch‐native ResNet implementation (18, 34, 50, 101, 152) without mmcv dependencies.  
  - Exposes `ResNetBackbone(depth=… , pretrained=True)` which returns feature maps at strides 4, 8, 16, 32.

- **`models/head/sep_aspp_contrast_head.py`**  
  - Implements a DeepLabV3+‐style head with Depthwise Separable ASPP and a contrastive “projection head” on the deepest feature.  
  - Returns `(logits, embedding)` where `logits` has shape `[B, total_classes, H/4, W/4]` and `embedding` has shape `[B, proj_dim, H/8, W/8]`.

- **`models/loss/`**  
  - **`cross_entropy_loss.py`**: a thin wrapper around PyTorch’s `CrossEntropyLoss` with `ignore_index`.  
  - **`tree_triplet_loss.py`**: helper that computes the triplet embedding term given a label hierarchy.  
  - **`hiera_triplet_loss.py`**: the two‐level (`fine→coarse`) BCE + Triplet loss (class consistency).  
  - **`rmi_hiera_triplet_loss.py`**: the three‐level (`fine→mid→high`) RMI + BCE + Triplet loss.

- **`train.py`**  
  - Reads a single YAML config and sets up:
    1. `HieroDataloader` for train/val  
    2. `ResNetBackbone` + `DepthwiseSeparableASPPContrastHead` + optional aux head  
    3. One of `HieraTripletLoss` (2‐level) or `RMIHieraTripletLoss` (3‐level) depending on whether super‐coarse maps exist in YAML  
    4. An SGD optimizer  
    5. Training & validation loops that compute loss + pixel accuracy and checkpoint when validation loss improves.
    6. Can also pass ``--pretrained`` to start training from a pretrained model

- **`infer.py`**
    - Loads a YAML config plus a ``.pth`` checkpoint and runs a single‐image inference:
        1. Parses command‐line arguments (``--config``, ``--checkpoint``, ``--image``, ``--device``)
        2. Reads the same YAML to build ``HieroDataloader``’s transforms (resize, normalize, etc.)
        3. Instantiates the backbone (e.g. ``ResNetBackbone``) and ``DepthwiseSeparableASPPContrastHead`` (with the correct output channels)
        4. Chooses the 2‐level (``HieraTripletLoss‐type``) or 3‐level (``RMIHieraTripletLoss‐type``) head structure based on whether the config has super‐coarse maps
        5. Loads model weights from the provided checkpoint onto the specified device (CPU or GPU)
        6. Preprocesses the input image (resize, normalize, convert to tensor), runs it through the network, and upsamples the logits to full resolution
        7. Applies ``argmax`` over the fine‐class logits (first ``n_fine`` channels) to produce a predicted mask, then maps back to coarse/super‐coarse if desired
        8. Saves or displays the resulting segmentation mask (e.g. as a PNG or overlaid on the original image)
        9. Optionally prints out per‐pixel class counts or an overall “pixel accuracy” if a ground‐truth mask is provided
        10. Allows switching between CUDA/CPU via the ``--device`` flag.
---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Shadowfear36/SegHiero
   cd SegHiero
2. **Create A Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate #linux/macOS
    venv\Scripts\activate # Windows
    ```
3. **Install Dependencies**
    ```bash
    pip install torch torchvision tqdm terminaltables pyyaml pillow numpy
    ```
    OR

    ```bash
    pip install -r requirements.txt
    ```


## Dataset & Label Preparation

SegHiero expects a simple directory structure with images and fine-level masks:

```
root/
    train/images/       # RGB images for training
    train/masks/        # Fine‐level masks (each pixel ∈ {0..n_fine−1} or 255 ignore)
    val/images/         # RGB images for validation
    val/masks/          # Fine‐level masks for validation
```

**Key points:**
- Fine-level masks use integer labels in `[0..n_fine−1]` and `255` for "ignore" pixels
- The hierarchical grouping (fine → coarse → super) is defined in your config file, not in the masks
- See [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md) for detailed instructions on creating masks, defining hierarchies, and troubleshooting
- See [tools/](tools/) for conversion scripts (COCO, Cityscapes, etc.)

---

## Architecture Configuration

SegHiero now supports multiple backbones and heads through a **factory pattern**. Simply edit your YAML config to switch architectures!

### Quick Examples

**ResNet + ASPP (Default, Proven):**
```yaml
backbone:
  type: "resnet"
  depth: 101
  pretrained: true

head:
  type: "aspp"
```

**SegFormer + SegFormer Head (Best Accuracy):**
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

**SegFormer + UltraFast Head (Speed/Accuracy Balance):**
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "ultrafast_segformer"
  proj_dim: 128
```

See [ARCHITECTURE_INTEGRATION_GUIDE.md](ARCHITECTURE_INTEGRATION_GUIDE.md) for comprehensive documentation and [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) for more configurations.

---

## Config File Format
Every training run is driven by a single YAML file. Below is a breakdown of each section.

## 1. **``dataset`` Section**
    ```yaml
    dataset:
    root: '/path/to/your/dataset'
    train:
      image_subdir: '/subdir'
      mask_subdir: '/subdir'
    val:
      image_subdir: '/subdir'
      mask_subdir: '/subdir'
    ```
    - ``root``: Base path to your dataset.
    - ``train.image_subdir``: Relative path under ``root`` for training images.
    - ``train.mask_subdir``:  Relative path under ``root`` for training masks.
    - ``val.image_subdir``, ``val.mask_subdir``: Analogous for validation.
## 2. **``classes`` Section**
- Defines your label hierarchy. Two variants:
    - **(a) Two‐Level Hierarchy (Fine→Coarse Only)**
        ```yaml
        classes:
            coarse_to_fine_map: [[0,3], [4,6], [7], [8]]
            coarse_names:
                0: Flower
                1: Tree
                2: Grass
                3: Mushroom
            fine_names:
                0: Sunflower
                1: Lily
                2: Rose
                3: Tulip
                4: Juniper
                5: Oak
                6: Palm
                7: Bermuda
                8: Lions Mane
        ```
        - ``coarse_to_fine_map``: A list of length = ``n_coarse``. Each element is ``[lbl]`` or ``[start,end]`` (inclusive).
        - ``coarse_names``: Mapping from coarse index → name.
        - ``fine_names``: Mapping from fine index → name, length = ``n_fine``.
        - When ``super_coarse_to_coarse_map`` is absent, the script uses two‐level loss (``HieraTripletLoss``).
     - **(B) Three‐Level Hierarchy (Fine→Coarse→Super)**
        ```yaml
        classes:
            super_coarse_to_coarse_map: [[0, 2], [3]]
            super_coarse_names:
                0: Plant
                1: Fungus
            coarse_to_fine_map: [[0,3], [4,6], [7], [8]]
            coarse_names:
                0: Flower
                1: Tree
                2: Grass
                3: Mushroom
            fine_names:
                0: Sunflower
                1: Lily
                2: Rose
                3: Tulip
                4: Juniper
                5: Oak
                6: Palm
                7: Bermuda
                8: Lions Mane
        ```
        - ``super_coarse_to_coarse_map``: Length = ``n_super``; each ``[lbl]`` or ``[start,end]``.
        - ``super_coarse_names``: Optional Mapping from super coarse index → name.
        - ``coarse_to_fine_map``, ``coarse_names``, ``fine_names``: As before.
        - When ``super_coarse_to_coarse_map`` exists, the script uses three‐level loss (``RMIHieraTripletLoss``).

## 3. **``backbone`` Section (NEW)**
    ```yaml
backbone:
  type: "resnet"           # Options: resnet, convnext, segformer
  depth: 101               # For ResNet: 18, 34, 50, 101, 152
  # variant: "tiny"        # For ConvNeXt: tiny, small, base, large, xlarge
  # variant: "mit-b0"      # For SegFormer: mit-b0, mit-b1, mit-b2, mit-b3, mit-b4, mit-b5
  pretrained: true         # Use ImageNet pretrained weights
    ```
    - ``type``: Backbone architecture to use
    - ``depth`` (ResNet only): Network depth
    - ``variant`` (ConvNeXt/SegFormer): Model variant
    - ``pretrained``: Whether to load ImageNet pretrained weights

## 4. **``head`` Section (NEW)**
    ```yaml
head:
  type: "aspp"             # Options: aspp, segformer, ultrafast_segformer, extremely_fast_segformer
  proj_dim: 256            # For SegFormer heads: embedding dimension (optional)
    ```
    - ``type``: Segmentation head architecture
    - ``proj_dim`` (SegFormer heads only): Projection dimension for embedding learning
      - Standard SegFormer: 256 (default)
      - UltraFast: 128 (default)
      - Extremely Fast: 64 (default)

## 5. **``model`` Section**
    ```yaml
model:
  pretrained_model: resnet-101
  c1_channels: 48
  aspp_channels: 512
  aspp_dilations: [1, 12, 24, 36]
  projection_dim: 256
  projection_type: "convmlp"
    ```
    - This section is now **only used for ASPP head configuration**
    - If using SegFormer heads, these parameters are ignored
    - The factory functions automatically handle channel dimensions based on your chosen backbone

## 6. **``training`` Section**
    ```yaml
        training:
            epochs: 50
            batch_size: 8
            lr: 0.001
            device: "cuda"
            fine_weight: 1.0
            coarse_weight: 1.0
            super_weight:   1.0 
            num_workers:  1
            gpus: [0]
    ```
    - The script sets ``CUDA_VISIBLE_DEVICES`` from ``gpus``.
    - If ``gpus: []``, runs on CPU.
## 7. **``transform`` Section**
    ```yaml
        transform:
            resize:     [150, 150]   # Resize each image & mask to 150×150
            hflip_prob: 0.5          # 50% chance to horizontally flip
    ```
    - When provided, ``HieroDataloader`` resizes both image and mask, then randomly flips with probability ``hflip_prob``.
    - If omitted, loader applies ``ToTensor()`` + normalization ``(0.485,0.456,0.406)/(0.229,0.224,0.225)``.

## 8. **``output`` Section**
    ```yaml
        output:
            checkpoint_dir: "./checkpoints"
            project_name:   "experiment_name"
    ```
    - ``checkpoint_dir``: Directory where the best model (as ``<project_name>_best.pth``) will be saved.
    - ``project_name``: Prefix for saved checkpoint filename.

## Training & Validation
- After preparing your dataset and writing a config file:
    ```bash
    python train.py --config /path/to/config.yaml --pretrained /path/to/pretrained/model #(<- optional)
    ```
- What happens:
    1. Dataset
        - Instantiates ``HieroDataloader`` for train/val.
        - Applies joint transforms (resize + flip + normalize) if specified.
    2. Hierarchy Setup
        - If ``super_coarse_to_coarse_map`` is present → three‐level loss; else → two‐level loss.
        - Builds appropriate mapping tensors / index lists
    3. Model
        - ``ResNetBackbone(depth=101, pretrained=True)`` → feature maps at strides 4, 8, 16, 32.
        - ``DepthwiseSeparableASPPContrastHead`` outputs ``(logits, embedding)``.
        - ``aux_head`` (1×1 conv + BN + ReLU) on C3 for fine classes.
    4. Loss
        - 2-Level: ``HieraTripletLoss(num_classes=n_fine, hiera_map, hiera_index, …)``
        - 3-Level: ``RMIHieraTripletLoss(n_fine, n_mid, n_high, fine_to_mid, fine_to_high, …)``
        - Both implement BCE‐style hierarchical consistency + cross‐entropy + triplet scheduling (cosine).
        - Auxiliary head uses standard ``CrossEntropyLoss(ignore_index=255)`` on fine masks.
    5. Optimizer
        - SGD over ``(backbone + main head + aux head)`` with lr, momentum=0.9, weight_decay=1e-4.
    6. Epoch Loop
        - Training: forward, compute ``main_loss + 0.4 * aux_loss``, backward, step.
        - Validation: evaluate same losses (no grad), accumulate pixel‐accuracy on fine predictions.
        - Print an ASCII table with ``[Epoch | Avg Train Loss | Avg Val Loss | Val Pixel Acc]``.
        - Save checkpoint if validation loss improves, under ``output.checkpoint_dir`` as ``<project_name>_best.pth``.

## Inference
- After Training, Pass your config file, along with an image, and your model to the infer.py file:
    ```bash
        python infer.py \
            --config ./config.yaml \
            --image ./samples/test_image.jpg \
            --checkpoint ./checkpoints/fun_best.pth \
            --device cuda:0 \
            --output-dir ./inference_outputs
    ```


## Custom Backbones / Heads

SegHiero uses a **factory pattern** that makes adding new architectures easy!

### Adding a New Backbone

1. Create your backbone file in `models/backbone/your_backbone.py`
2. Implement the standard interface (returns C1, C2, C3, C4 features)
3. Add import in `train.py`:
   ```python
   from models.backbone.your_backbone import YourBackbone
   ```
4. Add case to `create_backbone()` factory function (train.py:30-50):
   ```python
   elif backbone_type == 'your_backbone':
       return YourBackbone(...)
   ```
5. Add channel mapping to `get_backbone_channels()` (train.py:53-88):
   ```python
   elif backbone_type == 'your_backbone':
       return {'c1': ..., 'c2': ..., 'c3': ..., 'c4': ...}
   ```

### Adding a New Head

1. Create your head file in `models/head/your_head.py`
2. Implement the standard interface:
   - Input: List of four feature maps `[c1, c2, c3, c4]`
   - Output: `(logits, embedding)` tuple where:
     - `logits`: `[B, total_classes, H/4, W/4]`
     - `embedding`: `[B, D, H/8, W/8]` (for triplet loss)
3. Add import in `train.py`:
   ```python
   from models.head.your_head import YourHead
   ```
4. Add case to `create_head()` factory function (train.py:91-141):
   ```python
   elif head_type == 'your_head':
       return YourHead(...)
   ```

See [ARCHITECTURE_INTEGRATION_GUIDE.md](ARCHITECTURE_INTEGRATION_GUIDE.md) for detailed instructions.

### Requirements for Custom Architectures

- **Backbone**: Must return four feature maps at strides 4, 8, 16, 32.
- **Head**: Must accept a list of four feature maps and return `(logits, embedding)`
- **Loss Functions** (unchanged):
    - Two‐level: `HieraTripletLoss(num_classes, hiera_map, hiera_index, use_sigmoid=False, loss_weight=…)`
    - Three‐level: `RMIHieraTripletLoss(n_fine, n_mid, n_high, fine_to_mid, fine_to_high, ...)`
## License & Acknowledgements
**Inspired by**
- HieraSeg [https://github.com/lingorX/HieraSeg]

```
@article{li2022deep,
  title={Deep Hierarchical Semantic Segmentation},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Li, Jianwu and Yang, Yi},
  journal={arXiv preprint arXiv:2203.14335},
  year={2022}
}
```