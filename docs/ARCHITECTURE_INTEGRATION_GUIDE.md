# Architecture Integration Guide

This guide explains how to use different backbones and segmentation heads with SegHiero.

## Quick Start

The easiest way to change architectures is through the config file. Just update the `backbone` and `head` sections:

```yaml
backbone:
  type: "resnet"
  depth: 101
  pretrained: true

head:
  type: "aspp"
```

That's it. The training script handles the rest.

## Available Backbones

### ResNet (Default)

ResNet is the most stable option and works well for most use cases.

**Config:**
```yaml
backbone:
  type: "resnet"
  depth: 101        # Options: 18, 34, 50, 101, 152
  pretrained: true
```

**When to use:**
- You want proven, reliable performance
- You're training from scratch or fine-tuning
- You need good accuracy without much tuning

**Channel outputs:**
- ResNet-18/34: [64, 128, 256, 512]
- ResNet-50/101/152: [256, 512, 1024, 2048]

### ConvNeXt

ConvNeXt is a modern CNN architecture that often outperforms ResNet with similar speed.

**Config:**
```yaml
backbone:
  type: "convnext"
  variant: "tiny"   # Options: tiny, small, base, large, xlarge
  pretrained: true
```

**When to use:**
- You want better accuracy than ResNet
- You're okay with slightly slower training
- You have enough GPU memory (larger variants use more)

**Requirements:**
```bash
pip install timm>=0.9.0
```

**Channel outputs:**
- tiny/small: [96, 192, 384, 768]
- base: [128, 256, 512, 1024]
- large: [192, 384, 768, 1536]
- xlarge: [256, 512, 1024, 2048]

### SegFormer (Transformer-based)

SegFormer uses a hierarchical transformer backbone. Best results when paired with SegFormer heads.

**Config:**
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"  # Options: mit-b0 through mit-b5
  pretrained: true
```

**When to use:**
- You want state-of-the-art accuracy
- You're using a SegFormer head
- You have a good GPU (transformers are memory-hungry)

**Requirements:**
```bash
pip install transformers>=4.30.0
```

**Channel outputs:**
- mit-b0: [32, 64, 160, 256]
- mit-b1: [64, 128, 320, 512]
- mit-b2: [64, 128, 320, 512]
- mit-b3: [64, 128, 320, 512]
- mit-b4: [64, 128, 320, 512]
- mit-b5: [64, 128, 320, 512]

## Available Heads

### ASPP Head (Default)

The Atrous Spatial Pyramid Pooling head uses multiple dilation rates to capture multi-scale context. Works with any backbone.

**Config:**
```yaml
head:
  type: "aspp"
```

**When to use:**
- You're using ResNet or ConvNeXt backbone
- You want reliable, well-tested performance
- You need multi-scale feature aggregation

### SegFormer Head

The full SegFormer decoder. Best accuracy but slower than other options.

**Config:**
```yaml
head:
  type: "segformer"
  proj_dim: 256
```

**When to use:**
- You're using a SegFormer backbone
- Accuracy is more important than speed
- You have enough GPU memory

### UltraFast SegFormer Head

A lighter version of the SegFormer head. About 40% faster with minimal accuracy loss.

**Config:**
```yaml
head:
  type: "ultrafast_segformer"
  proj_dim: 128
```

**When to use:**
- You need faster training/inference
- You can accept ~1-2% accuracy drop
- You're memory-constrained

### Extremely Fast SegFormer Head

The lightest SegFormer variant. About 60% faster than the full version.

**Config:**
```yaml
head:
  type: "extremely_fast_segformer"
  proj_dim: 64
```

**When to use:**
- You need real-time or near-real-time performance
- You can accept ~3-5% accuracy drop
- You're very memory-constrained

## Recommended Combinations

### Maximum Accuracy
```yaml
backbone:
  type: "segformer"
  variant: "mit-b5"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

### Balanced Performance
```yaml
backbone:
  type: "resnet"
  depth: 101
  pretrained: true

head:
  type: "aspp"
```

### Maximum Speed
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "extremely_fast_segformer"
  proj_dim: 64
```

### Budget GPU (< 8GB VRAM)
```yaml
backbone:
  type: "resnet"
  depth: 50
  pretrained: true

head:
  type: "aspp"
```

## How It Works Under the Hood

The training script uses factory functions to create the right backbone and head based on your config:

1. `create_backbone(config)` - Creates the backbone network
2. `get_backbone_channels(config)` - Returns the channel dimensions for that backbone
3. `create_head(config, channels, num_classes)` - Creates the segmentation head

This means you can easily add new architectures by updating these three functions in `train.py`.

## Adding Your Own Backbone

If you want to add a new backbone:

1. Create a file in `models/backbone/your_backbone.py`
2. Make sure it returns 4 feature maps (C1, C2, C3, C4) at different resolutions
3. Add the import in `train.py`
4. Add a case to `create_backbone()` around line 29
5. Add channel dimensions to `get_backbone_channels()` around line 53

See CONTRIBUTING.md for detailed instructions.

## Adding Your Own Head

To add a new head:

1. Create a file in `models/head/your_head.py`
2. Make sure it takes `in_channels_list`, `num_classes`, and `proj_dim` as arguments
3. Return both `logits` and `embedding` from the forward pass
4. Add the import in `train.py`
5. Add a case to `create_head()` around line 91

See CONTRIBUTING.md for detailed instructions.

## Troubleshooting

**Out of memory errors:**
- Use a smaller backbone variant
- Use a faster head type
- Reduce batch size
- Use gradient checkpointing (requires code modification)

**Backbone not loading pretrained weights:**
- Check that you have the required dependencies installed
- Make sure `pretrained: true` is set in the config
- Check your internet connection (weights are downloaded on first use)

**Head incompatible with backbone:**
- All heads should work with all backbones
- If you get dimension mismatches, check `get_backbone_channels()` in train.py

**Training is very slow:**
- Try a smaller backbone (e.g., ResNet-50 instead of ResNet-101)
- Use a faster head variant
- Check that you're using GPU (`device: "cuda"` in config)

## Performance Notes

These are rough guidelines based on typical hardware (RTX 3090):

**Training speed (images/sec on single GPU):**
- ResNet-50 + ASPP: ~25 imgs/sec
- ResNet-101 + ASPP: ~18 imgs/sec
- ConvNeXt-tiny + ASPP: ~20 imgs/sec
- SegFormer-B0 + SegFormer: ~15 imgs/sec
- SegFormer-B5 + SegFormer: ~5 imgs/sec

**Memory usage (batch size 8):**
- ResNet-50 + ASPP: ~6GB
- ResNet-101 + ASPP: ~8GB
- ConvNeXt-base + ASPP: ~10GB
- SegFormer-B0 + SegFormer: ~7GB
- SegFormer-B5 + SegFormer: ~18GB

Your mileage will vary depending on input resolution, number of classes, and GPU.

## Questions?

Check the README.md for general usage or CONTRIBUTING.md if you want to add new architectures. Open an issue on GitHub if you run into problems.
