#!/usr/bin/env python3
"""
Improved Data Augmentation for U-Net Training

Adds strong augmentations to prevent overfitting and improve generalization.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import torch


class CryoEMAugmentation:
    """
    Strong augmentation pipeline for cryo-EM images

    Includes:
    - Geometric: rotation, flip, elastic deformation
    - Intensity: Gaussian noise, contrast adjustment
    - Domain-specific: simulated ice contamination
    """

    def __init__(self, p=0.5):
        """
        Args:
            p: probability of applying each augmentation
        """
        self.p = p

    def __call__(self, image, mask=None):
        """
        Apply augmentations to image and mask

        Args:
            image: numpy array [H, W]
            mask: numpy array [H, W] (optional)

        Returns:
            augmented_image, augmented_mask (if mask provided)
        """
        # Ensure float32
        image = image.astype(np.float32)
        if mask is not None:
            mask = mask.astype(np.float32)

        # 1. Random rotation
        if np.random.rand() < self.p:
            angle = np.random.uniform(-15, 15)
            image = self.rotate(image, angle)
            if mask is not None:
                mask = self.rotate(mask, angle, is_mask=True)

        # 2. Random flip
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                if mask is not None:
                    mask = np.fliplr(mask).copy()
            else:
                image = np.flipud(image).copy()
                if mask is not None:
                    mask = np.flipud(mask).copy()

        # 3. Elastic deformation
        if np.random.rand() < self.p * 0.5:  # Lower probability (expensive)
            image, mask = self.elastic_transform(image, mask, alpha=50, sigma=5)

        # 4. Gaussian noise
        if np.random.rand() < self.p:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*image.shape) * noise_std
            image = image + noise

        # 5. Contrast adjustment
        if np.random.rand() < self.p:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)   # Brightness
            image = alpha * image + beta

        # 6. Gaussian blur
        if np.random.rand() < self.p * 0.3:
            sigma = np.random.uniform(0.5, 1.5)
            image = gaussian_filter(image, sigma=sigma)

        # 7. Simulated ice contamination (domain-specific)
        if np.random.rand() < self.p * 0.2:
            image = self.add_ice_contamination(image)

        if mask is not None:
            return image, mask
        return image

    @staticmethod
    def rotate(image, angle, is_mask=False):
        """Rotate image by angle (degrees)"""
        h, w = image.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        if is_mask:
            # Use nearest neighbor for masks
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        else:
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
        return rotated

    @staticmethod
    def elastic_transform(image, mask, alpha=50, sigma=5):
        """Elastic deformation (like rubber sheet)"""
        shape = image.shape

        # Random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        # Mesh grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Apply deformation
        image_def = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        if mask is not None:
            mask_def = map_coordinates(mask, indices, order=0, mode='constant').reshape(shape)
            return image_def, mask_def

        return image_def, None

    @staticmethod
    def add_ice_contamination(image):
        """Simulate ice contamination (domain-specific for cryo-EM)"""
        h, w = image.shape

        # Random ice spots
        num_spots = np.random.randint(1, 5)
        for _ in range(num_spots):
            # Random location
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            radius = np.random.randint(10, 30)

            # Create mask
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
            ice_mask = dist_from_center <= radius

            # Add bright spot (ice)
            image[ice_mask] += np.random.uniform(0.2, 0.5)

        return image


class ToTensor:
    """Convert numpy arrays to PyTorch tensors"""

    def __call__(self, image, mask=None):
        # Add channel dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
            return image_tensor, mask_tensor

        return image_tensor


class Normalize:
    """Normalize image to zero mean and unit variance"""

    def __call__(self, image, mask=None):
        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-8)

        if mask is not None:
            return image, mask
        return image


class Compose:
    """Compose multiple transformations"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            if mask is not None:
                image, mask = t(image, mask)
            else:
                image = t(image)

        if mask is not None:
            return image, mask
        return image


# ============================================================================
# Visualization
# ============================================================================

def visualize_augmentations(image, mask=None, num_examples=6):
    """Visualize augmentation examples"""
    import matplotlib.pyplot as plt

    aug = CryoEMAugmentation(p=0.8)

    fig, axes = plt.subplots(2, num_examples, figsize=(15, 5))

    for i in range(num_examples):
        if mask is not None:
            aug_img, aug_mask = aug(image.copy(), mask.copy())
            axes[0, i].imshow(aug_img, cmap='gray')
            axes[1, i].imshow(aug_mask, cmap='hot')
        else:
            aug_img = aug(image.copy())
            axes[0, i].imshow(aug_img, cmap='gray')
            axes[1, i].axis('off')

        axes[0, i].set_title(f'Aug {i+1}')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=12)
    if mask is not None:
        axes[1, 0].set_ylabel('Mask', fontsize=12)

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: augmentation_examples.png")
    plt.close()


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == '__main__':
    print("Testing augmentation pipeline...\n")

    # Load sample image
    import sys
    from pathlib import Path

    test_images = list(Path('data/unet_test_heldout/images').glob('*.npy'))

    if len(test_images) > 0:
        print(f"Loading sample image: {test_images[0]}")
        image = np.load(test_images[0])

        # Create dummy mask
        mask = np.zeros_like(image)
        h, w = image.shape
        cv2.circle(mask, (w//2, h//2), 15, 1, -1)

        # Visualize
        visualize_augmentations(image, mask, num_examples=6)
    else:
        print("No test images found. Using random image.")
        image = np.random.randn(128, 128)
        visualize_augmentations(image, num_examples=6)

    print("\n" + "="*80)
    print("USAGE IN TRAINING:")
    print("="*80)
    print("""
# In your training script:

from improved_augmentation import CryoEMAugmentation, Normalize, ToTensor, Compose

# Training augmentation (strong)
train_transform = Compose([
    CryoEMAugmentation(p=0.5),  # Apply augmentations with 50% probability
    Normalize(),                 # Normalize
    ToTensor()                   # Convert to tensor
])

# Validation/test (no augmentation, only normalize)
val_transform = Compose([
    Normalize(),
    ToTensor()
])

# In dataset __getitem__:
if self.split == 'train':
    image, mask = train_transform(image, mask)
else:
    image, mask = val_transform(image, mask)
""")
