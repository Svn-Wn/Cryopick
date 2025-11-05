#!/usr/bin/env python3
"""
Improved Loss Functions for Better U-Net Performance

Replace standard BCE with better losses:
1. Focal Loss - Focus on hard examples
2. Dice Loss - Better for segmentation
3. Combined Loss - Best of both worlds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - Focuses on hard-to-classify examples

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -α(1-p_t)^γ log(p_t)

    where:
        p_t = p if y=1, else 1-p
        α = balancing factor (default: 0.25)
        γ = focusing parameter (default: 2.0)

    Benefits:
    - Reduces loss contribution from easy examples
    - Focuses training on hard negatives/positives
    - Better for imbalanced data
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets, ignore_mask=None):
        """
        Args:
            inputs: predicted logits [B, 1, H, W]
            targets: ground truth [B, 1, H, W] or [B, H, W]
            ignore_mask: optional mask (-1 = ignore) [B, 1, H, W]
        """
        # Ensure same shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Convert targets to float (important for loss computation)
        targets = targets.float()

        # Create valid mask from targets (ignore -1)
        if ignore_mask is None:
            valid_mask = (targets != -1).float()
        else:
            if ignore_mask.dim() == 3:
                ignore_mask = ignore_mask.unsqueeze(1)
            valid_mask = (ignore_mask != -1).float()

        # Clamp targets to [0, 1] for loss computation
        targets_clamped = targets.clamp(0, 1)

        # BCE with logits (with optional pos_weight)
        if self.pos_weight is not None:
            # Convert pos_weight to tensor if needed
            if isinstance(self.pos_weight, (int, float)):
                pos_weight_tensor = torch.tensor([self.pos_weight], device=inputs.device)
            else:
                pos_weight_tensor = self.pos_weight
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets_clamped, pos_weight=pos_weight_tensor, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_clamped, reduction='none')

        # Probability
        probs = torch.sigmoid(inputs)

        # p_t
        p_t = probs * targets_clamped + (1 - probs) * (1 - targets_clamped)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets_clamped + (1 - self.alpha) * (1 - targets_clamped)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply valid mask (ignore -1 pixels)
        focal_loss = focal_loss * valid_mask

        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - Directly optimizes Dice coefficient (F1-score)

    Dice = 2*|X∩Y| / (|X|+|Y|)
    Loss = 1 - Dice

    Benefits:
    - Better for segmentation tasks
    - Handles class imbalance naturally
    - Directly optimizes overlap
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, ignore_mask=None):
        """
        Args:
            inputs: predicted logits [B, 1, H, W]
            targets: ground truth [B, 1, H, W]
            ignore_mask: optional mask (-1 = ignore)
        """
        # Sigmoid activation
        probs = torch.sigmoid(inputs)

        # Ensure same shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Convert targets to float (important for loss computation)
        targets = targets.float()

        # Create valid mask from targets (ignore -1)
        if ignore_mask is None:
            valid_mask = (targets != -1).float()
        else:
            if ignore_mask.dim() == 3:
                ignore_mask = ignore_mask.unsqueeze(1)
            valid_mask = (ignore_mask != -1).float()

        # Clamp targets to [0, 1]
        targets_clamped = targets.clamp(0, 1)

        # Apply mask to predictions and targets
        probs_masked = probs * valid_mask
        targets_masked = targets_clamped * valid_mask

        # Flatten
        probs_flat = probs_masked.view(-1)
        targets_flat = targets_masked.view(-1)

        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        # Dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Loss = Focal Loss + Dice Loss

    Benefits:
    - Focal: handles hard examples and class imbalance
    - Dice: optimizes segmentation overlap
    - Combined: best of both worlds

    Recommended weights:
    - focal_weight=0.7, dice_weight=0.3 (default)
    - focal_weight=0.5, dice_weight=0.5 (balanced)
    """

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, pos_weight=None,
                 focal_weight=0.7, dice_weight=0.3, smooth=1.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, ignore_mask=None):
        """
        Args:
            inputs: predicted logits [B, 1, H, W]
            targets: ground truth [B, 1, H, W]
            ignore_mask: optional mask (-1 = ignore)
        """
        focal = self.focal_loss(inputs, targets, ignore_mask)
        dice = self.dice_loss(inputs, targets, ignore_mask)

        combined = self.focal_weight * focal + self.dice_weight * dice

        return combined, {'focal': focal.item(), 'dice': dice.item(), 'total': combined.item()}


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice loss with adjustable FP/FN penalty

    Benefits:
    - Control trade-off between precision and recall
    - α > β: penalize FP more (higher precision)
    - α < β: penalize FN more (higher recall)
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        Args:
            alpha: weight for false positives
            beta: weight for false negatives
            alpha=beta=0.5: equivalent to Dice loss
            alpha=0.7, beta=0.3: favor precision
            alpha=0.3, beta=0.7: favor recall
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets, ignore_mask=None):
        probs = torch.sigmoid(inputs)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Convert targets to float
        targets = targets.float()

        if ignore_mask is not None:
            if ignore_mask.dim() == 3:
                ignore_mask = ignore_mask.unsqueeze(1)
            mask = (ignore_mask != -1).float()
            probs = probs * mask
            targets = targets * mask

        # True Positives, False Positives, False Negatives
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - tversky


# ============================================================================
# Usage Example
# ============================================================================

def compare_losses_example():
    """Example showing how different losses behave"""

    import numpy as np
    import matplotlib.pyplot as plt

    # Create sample predictions and targets
    batch_size = 4
    height, width = 128, 128

    # Simulated predictions (logits)
    predictions = torch.randn(batch_size, 1, height, width)

    # Simulated ground truth
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    # Initialize losses
    bce_loss = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()

    # Compute losses
    bce = bce_loss(predictions, targets)
    focal = focal_loss(predictions, targets)
    dice = dice_loss(predictions, targets)
    combined, components = combined_loss(predictions, targets)

    print("Loss Comparison:")
    print(f"  BCE:       {bce.item():.4f}")
    print(f"  Focal:     {focal.item():.4f}")
    print(f"  Dice:      {dice.item():.4f}")
    print(f"  Combined:  {combined.item():.4f}")
    print(f"    - Focal component: {components['focal']:.4f}")
    print(f"    - Dice component:  {components['dice']:.4f}")


if __name__ == '__main__':
    print("Testing improved loss functions...\n")
    compare_losses_example()

    print("\n" + "="*80)
    print("USAGE IN TRAINING:")
    print("="*80)
    print("""
# In your training script, replace:

# OLD:
criterion = nn.BCEWithLogitsLoss()

# NEW (Recommended):
from improved_losses import CombinedLoss
criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)

# Training loop:
outputs = model(images)
loss, components = criterion(outputs, masks, ignore_mask)

# Log components for monitoring
print(f"Focal: {components['focal']:.4f}, Dice: {components['dice']:.4f}")
""")
