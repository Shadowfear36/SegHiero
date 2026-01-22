# Contributing to SegHiero

Thank you for your interest in contributing to SegHiero! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Your environment (Python version, PyTorch version, OS, GPU)
- Relevant config files and error messages

### Suggesting Features

We love new ideas! Please create an issue describing:
- The feature and its use case
- Why it would be valuable to the community
- Possible implementation approach (if you have one)

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update documentation if needed

3. **Test your changes**:
   - Run training for at least 1 epoch to verify it works
   - Test with different backbone/head combinations if applicable

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with a clear description of your changes.

---

## 🏗️ Adding New Architectures

### Adding a New Backbone

SegHiero makes it easy to add new backbones thanks to the factory pattern!

**1. Create your backbone file:**
```python
# models/backbone/your_backbone.py

import torch.nn as nn

class YourBackbone(nn.Module):
    def __init__(self, variant='base', pretrained=True):
        super().__init__()
        # Your initialization code
        ...

    def forward(self, x):
        # Must return 4 feature maps at different resolutions
        c1 = ...  # 1/4 resolution
        c2 = ...  # 1/8 resolution
        c3 = ...  # 1/16 resolution
        c4 = ...  # 1/32 resolution
        return c1, c2, c3, c4
```

**2. Update `train.py`:**

Add import (around line 15):
```python
from models.backbone.your_backbone import YourBackbone
```

Add to `create_backbone()` function (around line 30):
```python
elif backbone_type == 'your_backbone':
    variant = backbone_config.get('variant', 'base')
    pretrained = backbone_config.get('pretrained', True)
    return YourBackbone(variant=variant, pretrained=pretrained)
```

Add to `get_backbone_channels()` function (around line 53):
```python
elif backbone_type == 'your_backbone':
    variant = backbone_config.get('variant', 'base')
    channels_map = {
        'base': {'c1': 64, 'c2': 128, 'c3': 256, 'c4': 512},
        # Add other variants...
    }
    return channels_map.get(variant, channels_map['base'])
```

**3. Test it:**
```yaml
backbone:
  type: "your_backbone"
  variant: "base"
  pretrained: true
```

### Adding a New Head

**1. Create your head file:**
```python
# models/head/your_head.py

import torch.nn as nn

class YourHead(nn.Module):
    def __init__(self, in_channels_list, num_classes, proj_dim=256):
        super().__init__()
        # in_channels_list = [c1_channels, c2_channels, c3_channels, c4_channels]
        # Your initialization code
        ...

    def forward(self, features):
        # features = [c1, c2, c3, c4]
        c1, c2, c3, c4 = features

        # Your forward pass
        logits = ...     # [B, num_classes, H, W]
        embedding = ...  # [B, proj_dim, H', W'] for triplet loss

        return logits, embedding
```

**2. Update `train.py`:**

Add import (around line 20):
```python
from models.head.your_head import YourHead
```

Add to `create_head()` function (around line 91):
```python
elif head_type == 'your_head':
    in_channels_list = [in_channels_dict['c1'], in_channels_dict['c2'],
                       in_channels_dict['c3'], in_channels_dict['c4']]
    return YourHead(
        in_channels_list=in_channels_list,
        num_classes=num_classes,
        proj_dim=head_config.get('proj_dim', 256)
    )
```

**3. Test it:**
```yaml
head:
  type: "your_head"
  proj_dim: 256
```

---

## 📝 Code Style Guidelines

- Use descriptive variable names
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and modular
- Add comments for complex logic

**Example docstring:**
```python
def create_backbone(config):
    """
    Factory function to create backbone based on config.

    Args:
        config (dict): Full configuration dictionary containing 'backbone' section

    Returns:
        nn.Module: Initialized backbone network

    Raises:
        ValueError: If backbone type is not recognized
    """
    ...
```

---

## 🧪 Testing

Before submitting a PR, please test:

1. **Basic training** (1 epoch is fine):
   ```bash
   python train.py --config test-config.yaml
   ```

2. **Different architecture combinations**:
   - Your new backbone with ASPP head
   - Your new head with ResNet backbone
   - Full integration if you added both

3. **Inference**:
   ```bash
   python infer.py --config config.yaml --image test.jpg --checkpoint model.pth
   ```

---

## 📖 Documentation

If you add a new feature, please update:

1. **README.md** - Add to the relevant section
2. **ARCHITECTURE_INTEGRATION_GUIDE.md** - If adding new backbone/head
3. **CONFIG_EXAMPLES.md** - Add example configuration
4. **Docstrings** - In your code
5. **example-config.yaml** - Add commented example if relevant

---

## 🎯 Good First Issues

Looking for a place to start? Check issues labeled `good-first-issue`:

- Documentation improvements
- Adding new backbone/head architectures
- Performance benchmarking
- Bug fixes

---

## 💬 Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Email**: Contact the maintainers directly

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make SegHiero better! 🚀
