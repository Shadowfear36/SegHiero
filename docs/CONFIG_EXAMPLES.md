# Configuration Examples

Quick reference for different architecture configurations.

## 🎯 Copy-Paste Configurations

### ResNet + ASPP (Default, Proven)

```yaml
backbone:
  type: "resnet"
  depth: 101
  pretrained: true

head:
  type: "aspp"

model:
  c1_channels: 48
  aspp_channels: 512
  aspp_dilations: [1, 12, 24, 36]
  projection_dim: 256
  projection_type: "convmlp"
```

---

### SegFormer B0 + SegFormer Head (Best Accuracy)

```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256

# model section not needed for SegFormer heads
```

---

### SegFormer B0 + UltraFast Head (Speed/Accuracy Balance)

```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "ultrafast_segformer"
  proj_dim: 128
```

---

### SegFormer B0 + Extremely Fast Head (Maximum Speed)

```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "extremely_fast_segformer"
  proj_dim: 64
```

---

### ConvNeXt Tiny + ASPP

```yaml
backbone:
  type: "convnext"
  variant: "tiny"
  pretrained: true

head:
  type: "aspp"

model:
  c1_channels: 48
  aspp_channels: 512
  aspp_dilations: [1, 12, 24, 36]
  projection_dim: 256
  projection_type: "convmlp"
```

---

### ConvNeXt Tiny + SegFormer Head

```yaml
backbone:
  type: "convnext"
  variant: "tiny"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

---

### ResNet-50 + UltraFast SegFormer Head

```yaml
backbone:
  type: "resnet"
  depth: 50
  pretrained: true

head:
  type: "ultrafast_segformer"
  proj_dim: 128
```

---

### SegFormer B2 + SegFormer Head (High Accuracy, Slower)

```yaml
backbone:
  type: "segformer"
  variant: "mit-b2"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

---

## 📋 All Backbone Options

### ResNet
```yaml
backbone:
  type: "resnet"
  depth: [18, 34, 50, 101, 152]  # Choose one
  pretrained: true
```

### ConvNeXt
```yaml
backbone:
  type: "convnext"
  variant: [tiny, small, base, large, xlarge]  # Choose one
  pretrained: true
```

### SegFormer
```yaml
backbone:
  type: "segformer"
  variant: [mit-b0, mit-b1, mit-b2, mit-b3, mit-b4, mit-b5]  # Choose one
  pretrained: true
```

---

## 📋 All Head Options

### ASPP Head
```yaml
head:
  type: "aspp"

# Requires model section:
model:
  c1_channels: 48
  aspp_channels: 512
  aspp_dilations: [1, 12, 24, 36]
  projection_dim: 256
  projection_type: "convmlp"
```

### Standard SegFormer Head
```yaml
head:
  type: "segformer"
  proj_dim: 256  # Default: 256
```

### UltraFast SegFormer Head
```yaml
head:
  type: "ultrafast_segformer"
  proj_dim: 128  # Default: 128, can use 256 for higher accuracy
```

### Extremely Fast SegFormer Head
```yaml
head:
  type: "extremely_fast_segformer"
  proj_dim: 64  # Default: 64, can use 128 for higher accuracy
```

---

## 🚀 Performance Tiers

### Maximum Accuracy (Slowest)
```yaml
backbone:
  type: "segformer"
  variant: "mit-b5"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

### High Accuracy (Balanced)
```yaml
backbone:
  type: "segformer"
  variant: "mit-b2"
  pretrained: true

head:
  type: "segformer"
  proj_dim: 256
```

### Good Accuracy, Fast
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "ultrafast_segformer"
  proj_dim: 128
```

### Maximum Speed (Slight Accuracy Loss)
```yaml
backbone:
  type: "segformer"
  variant: "mit-b0"
  pretrained: true

head:
  type: "extremely_fast_segformer"
  proj_dim: 64
```

---

## 💾 Full Example Config

See `example-config.yaml` for a complete configuration file with all options documented.

---

## 🔄 Migrating Existing Configs

If you have an old config with only:
```yaml
model:
  backbone_depth: 101
```

Just add these sections to enable new architectures:
```yaml
backbone:
  type: "resnet"
  depth: 101
  pretrained: true

head:
  type: "aspp"
```

Your existing `model:` section will still be used for ASPP head parameters!
