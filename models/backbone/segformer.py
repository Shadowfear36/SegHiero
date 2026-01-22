import torch
import torch.nn as nn
from transformers import SegformerModel
from typing import List, Tuple

class SegFormerBackbone(nn.Module):
    """
    Simple SegFormer backbone that maps features correctly to expected channels
    """
    
    def __init__(self, variant='mit-b0', pretrained=True, **kwargs):
        super().__init__()
        
        # SegFormer model configurations
        model_configs = {
            'mit-b0': {
                'model_name': 'nvidia/mit-b0',
                'channels': [32, 64, 160, 256],
                'params': '3.7M'
            },
            'mit-b1': {
                'model_name': 'nvidia/mit-b1', 
                'channels': [64, 128, 320, 512],
                'params': '14M'
            },
            'mit-b2': {
                'model_name': 'nvidia/mit-b2',
                'channels': [64, 128, 320, 512], 
                'params': '25M'
            },
            'mit-b3': {
                'model_name': 'nvidia/mit-b3',
                'channels': [64, 128, 320, 512],
                'params': '45M'
            },
            'mit-b4': {
                'model_name': 'nvidia/mit-b4',
                'channels': [64, 128, 320, 512],
                'params': '62M'
            },
            'mit-b5': {
                'model_name': 'nvidia/mit-b5',
                'channels': [64, 128, 320, 512],
                'params': '82M'
            }
        }
        
        if variant not in model_configs:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(model_configs.keys())}")
        
        config = model_configs[variant]
        self.variant = variant
        self.out_channels = config['channels']
        
        print(f"→ Loading SegFormer-{variant} ({config['params']} parameters)")
        
        if pretrained:
            try:
                # Load pretrained model
                self.segformer = SegformerModel.from_pretrained(
                    config['model_name'],
                    output_hidden_states=True,
                    return_dict=True
                )
                print(f"→ Loaded pretrained weights from {config['model_name']}")
            except Exception as e:
                print(f"→ Warning: Could not load pretrained weights: {e}")
                print(f"→ Using random initialization")
                from transformers import SegformerConfig
                segformer_config = SegformerConfig.from_pretrained(config['model_name'])
                self.segformer = SegformerModel(segformer_config)
        else:
            # Load config only, no pretrained weights
            from transformers import SegformerConfig
            segformer_config = SegformerConfig.from_pretrained(config['model_name'])
            self.segformer = SegformerModel(segformer_config)
            print(f"→ Using random initialization for SegFormer-{variant}")
    
    def forward(self, x):
        """
        Simple forward pass that outputs features with correct channel dimensions
        """
        # Get multi-scale features from SegFormer encoder
        outputs = self.segformer(x, output_hidden_states=True)
        
        # Extract the 3 SegFormer hidden states (skip the first one which is input embeddings)
        hidden_states = outputs.hidden_states[1:]  # Should give us 3 feature maps
        
        if len(hidden_states) != 3:
            raise ValueError(f"Expected 3 SegFormer features, got {len(hidden_states)}")
        
        # SegFormer features are already in spatial format [B, C, H, W]
        seg_f1, seg_f2, seg_f3 = hidden_states
        
        # print(f"DEBUG: SegFormer raw features - f1: {seg_f1.shape}, f2: {seg_f2.shape}, f3: {seg_f3.shape}")
        
        # Map SegFormer features to expected output channels:
        # seg_f1 → c2 (keep original channels)
        # seg_f2 → c3 (keep original channels) 
        # seg_f3 → c4 (keep original channels)
        # Create c1 by upsampling c2
        
        c2 = seg_f1  # Should be [B, 64, H/8, W/8]
        c3 = seg_f2  # Should be [B, 160, H/16, W/16]  
        c4 = seg_f3  # Should be [B, 256, H/32, W/32]
        
        # Create c1 by upsampling c2 and reducing channels
        c1 = nn.functional.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Add channel adaptation layers if they don't exist
        if not hasattr(self, 'c1_adapter'):
            self.c1_adapter = nn.Conv2d(c1.shape[1], self.out_channels[0], 1).to(x.device)  # 64 → 32
            print(f"→ Created c1 adapter: {c1.shape[1]} → {self.out_channels[0]} channels")
        
        # Apply channel adaptation for c1 only (others should already match)
        c1 = self.c1_adapter(c1)
        
        # print(f"DEBUG: Final features - c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}")
        
        # Verify channel dimensions match expected
        expected_channels = self.out_channels
        actual_channels = [c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1]]
        
        if actual_channels != expected_channels:
            print(f"WARNING: Channel mismatch!")
            print(f"  Expected: {expected_channels}")
            print(f"  Actual: {actual_channels}")
        
        return c1, c2, c3, c4
    
    def get_channels(self):
        """Return channel dimensions for each feature level"""
        return {
            'c1': self.out_channels[0],
            'c2': self.out_channels[1], 
            'c3': self.out_channels[2],
            'c4': self.out_channels[3]
        }


# Convenience functions for easy instantiation
def segformer_b0(pretrained=True, **kwargs):
    """SegFormer-B0: Ultra-fast, 3.7M parameters"""
    return SegFormerBackbone('mit-b0', pretrained=pretrained, **kwargs)

def segformer_b1(pretrained=True, **kwargs):
    """SegFormer-B1: Balanced speed/accuracy, 14M parameters"""
    return SegFormerBackbone('mit-b1', pretrained=pretrained, **kwargs)

def segformer_b2(pretrained=True, **kwargs):
    """SegFormer-B2: Higher accuracy, 25M parameters"""
    return SegFormerBackbone('mit-b2', pretrained=pretrained, **kwargs)