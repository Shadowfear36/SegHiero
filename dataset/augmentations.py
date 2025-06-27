import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
from PIL import Image, ImageFilter, ImageEnhance

class MotionBlurTransform:
    def __init__(self, p=0.3, max_kernel_size=7):
        self.p = p
        self.max_kernel_size = max_kernel_size
    
    def __call__(self, image):
        if random.random() < self.p:
            kernel_size = random.randint(3, self.max_kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Create horizontal motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # Apply blur
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
        return image

class DustSpotTransform:
    def __init__(self, p=0.2, max_spots=5):
        self.p = p
        self.max_spots = max_spots
    
    def __call__(self, image):
        if random.random() < self.p:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            num_spots = random.randint(1, self.max_spots)
            for _ in range(num_spots):
                # Random spot location and size
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                radius = random.randint(2, 8)
                
                # Create a mask for the circle
                Y, X = np.ogrid[:h, :w]
                mask = (X - x)**2 + (Y - y)**2 <= radius**2
                
                # Apply darkening to the masked region
                img_array[mask] = (img_array[mask] * 0.7).astype(np.uint8)
            
            return Image.fromarray(img_array)
        return image

class ShadowTransform:
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, image):
        if random.random() < self.p:
            img_array = np.array(image).astype(np.float32)
            h, w = img_array.shape[:2]
            
            # Create gradient shadow
            shadow_strength = random.uniform(0.7, 0.9)
            gradient = np.linspace(shadow_strength, 1.0, w)
            gradient = np.tile(gradient, (h, 1))
            
            # Apply shadow
            img_array = img_array * gradient[..., np.newaxis]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
        return image

class GaussianNoiseTransform:
    def __init__(self, std=0.01, p=0.2):
        self.std = std
        self.p = p
    
    def __call__(self, image):
        if random.random() < self.p:
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.std, img_array.shape)
            noisy = img_array + noise
            noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)
        return image

class ProductionJointTransform:
    """
    Joint transform that combines your existing JointTransform with production augmentations.
    Applies production augmentations to image only (not mask), then does geometric transforms jointly.
    """
    def __init__(self, resize=None, hflip_prob=0.5,
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225),
                 production_aug_prob=0.4,
                 augmentation_strength=1.0):
        
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.production_aug_prob = production_aug_prob
        self.strength = augmentation_strength
        
        # Production-specific augmentations (image only)
        self.production_transforms = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3 * self.strength,
                contrast=0.2 * self.strength,
                saturation=0.1 * self.strength,
                hue=0.05 * self.strength
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=1 + self.strength, 
                p=0.3
            ),
            MotionBlurTransform(p=0.3 * self.strength),
            DustSpotTransform(p=0.2 * self.strength),
            ShadowTransform(p=0.3 * self.strength),
            GaussianNoiseTransform(std=0.01 * self.strength, p=0.2),
        ])

    def __call__(self, img: Image.Image, mask: torch.Tensor):
        # 1) Apply production augmentations to image only (probabilistically)
        if random.random() < self.production_aug_prob:
            img = self.production_transforms(img)
        
        # 2) Apply geometric transforms (resize + flip) to both image and mask
        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=self.resize,
                mode="nearest"
            ).long().squeeze(0).squeeze(0)
        
        if torch.rand(1).item() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = torch.flip(mask, dims=[1])
        
        # 3) Convert to tensor and normalize
        img_t = transforms.ToTensor()(img)
        img_t = transforms.Normalize(mean=self.normalize_mean,
                                   std=self.normalize_std)(img_t)
        
        return img_t, mask