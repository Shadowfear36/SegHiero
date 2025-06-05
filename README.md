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

## Table of Contents

1. [What Is Hierarchical Semantic Segmentation?](#what-is-hierarchical-semantic-segmentation)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Dataset & Label Preparation](#dataset--label-preparation)  
5. [Config File Format](#config-file-format)  
   - [1. Dataset Section](#1-dataset-section)  
   - [2. Classes Section](#2-classes-section)  
   - [3. Model Section](#3-model-section)  
   - [4. Training Section](#4-training-section)  
   - [5. Transform Section](#5-transform-section)  
   - [6. Output Section](#6-output-section)  
6. [Training & Validation](#training--validation)  
7. [Inference](#inference)  
8. [Custom Backbones / Heads](#custom-backbones--heads)  
9. [License & Acknowledgments](#license--acknowledgments)

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
│ └── dataloader.py # HieroDataloader implementation
│
├── models/
│ ├── backbone/
│ │ └── resnet.py # ResNetBackbone (vanilla PyTorch)
│ ├── head/
│ │ └── sep_aspp_contrast_head.py # DepthwiseSeparableASPPContrastHead
│ └── loss/
│ ├── cross_entropy_loss.py # Wrapper around torch.nn.CrossEntropyLoss
│ ├── tree_triplet_loss.py # Triplet‐loss for hierarchical embeddings
│ ├── hiera_triplet_loss.py # 2‐level HieraTripletLoss
│ └── rmi_hiera_triplet_loss.py # 3‐level RMIHieraTripletLoss
│
├── train.py # Main training & validation script, consumes a single YAML config
└── example-config.yaml # Example hierarchy config
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
- **SegHiero Expects:**
    - A root Directory Structured like:
        ```
        root/
            train/images/       # RGB images for training
            train/masks/        # “Fine‐level” masks (each pixel ∈ {0..n_fine−1} or 255 ignore)
            val/images/         # RGB images for validation
            val/masks/          # Fine‐level masks for validation
        ```
    - **Fine‐level masks** must use integer labels in ``[0..n_fine−1]`` and ``255`` for “ignore” pixels.

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

## 3. **``model`` Section**
    ```yaml
        model:
            pretrained_model: resnet-101
    ```
    Currently strictly informational; by default, ``train.py`` uses ``ResNetBackbone(depth=101, pretrained=True)``. In future update it will allow for choice of multiple backbones.

## 4. **``training`` Section**
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
## 5. **``transform`` Section**
    ```yaml
        transform:
            resize:     [150, 150]   # Resize each image & mask to 150×150
            hflip_prob: 0.5          # 50% chance to horizontally flip
    ```
    - When provided, ``HieroDataloader`` resizes both image and mask, then randomly flips with probability ``hflip_prob``.
    - If omitted, loader applies ``ToTensor()`` + normalization ``(0.485,0.456,0.406)/(0.229,0.224,0.225)``.

## 6. **``output`` Section**
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
    python train.py --config /path/to/config.yaml
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
- You can swap in any backbone or head that matches SegHiero’s expectations:
    - **Backbone**: Must return four feature maps at strides 4, 8, 16, 32.
        E.g. replace ``ResNetBackbone`` in train.py (lines 115–118) with ``UNet`` or ``HRNet``.
    - **Head**: Must accept a list of four feature maps and return ``(logits, embedding)``:
        - ``logits``: ``[B, total_classes, H/4, W/4]``
        - ``embedding``: ``[B, D, H/8, W/8]`` (for triplet loss) If your head does not produce embeddings, you can modify the loss to ignore triplet scheduling.
    - **Loss Functions**:
        - Two‐level: ``HieraTripletLoss(num_classes, hiera_map, hiera_index, use_sigmoid=False, loss_weight=…)``
        - Three‐level: ``RMIHieraTripletLoss(n_fine, n_mid, n_high, fine_to_mid, fine_to_high, rmi_radius, rmi_pool_way, rmi_pool_size, rmi_pool_stride, loss_weight_lambda, loss_weight, ignore_index)``
## License & Acknowledgements