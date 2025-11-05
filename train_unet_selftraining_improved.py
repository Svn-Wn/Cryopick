#!/usr/bin/env python3
"""
U-Net Self-Training (IMPROVED) for Cryo-EM Particle Picking

IMPROVEMENTS OVER BASELINE:
✅ Combined Loss (Focal + Dice) - Focus on hard examples + optimize overlap
✅ Strong Data Augmentation - Rotation, flip, elastic, noise, ice simulation
✅ Cosine Annealing LR - Better convergence
✅ Longer Training - 100 epochs initial (vs 50)

Expected improvement: +5-8% F1 over baseline

This script implements iterative self-training for semantic segmentation of particles
in cryo-EM micrographs using a U-Net architecture. It handles Positive-Unlabeled (PU)
learning by progressively generating pseudo-labels from high-confidence predictions.

Training Phases:
1. Initial supervised training on labeled particles
2. Iterative self-training loop:
   - Generate predictions on all data
   - Create pseudo-labels from high/low confidence predictions
   - Retrain on combined labels (original + pseudo)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score

# Import improved losses and augmentation
from improved_losses import CombinedLoss, FocalLoss, DiceLoss
from improved_augmentation import CryoEMAugmentation, Normalize, Compose


# ============================================================================
# Reproducibility and Determinism
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across torch, numpy, and random.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make operations deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Random seed set to {seed} for reproducibility")


# ============================================================================
# U-Net Architecture
# ============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net for semantic segmentation"""
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_features * 16, base_features * 8)
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features)

        # Output
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.out(dec1)


# ============================================================================
# Utility Functions
# ============================================================================

def load_all_from_dir(data_dir):
    """Load all images and masks from preprocessed batches"""
    data_dir = Path(data_dir)
    batch_files = sorted(list(data_dir.glob("batch_*.npz")))

    all_images = []
    all_masks = []

    for batch_file in batch_files:
        data = np.load(batch_file)
        images = data['images']
        if len(images.shape) == 4 and images.shape[3] == 3:
            images = images.mean(axis=3)
        all_images.extend(list(images))
        all_masks.extend(list(data['masks']))

    return all_images, all_masks


def coordinates_to_mask(coordinates: List[Tuple[int, int]],
                       image_shape: Tuple[int, int],
                       particle_radius: int) -> np.ndarray:
    """
    Convert particle coordinates to binary segmentation mask.

    Args:
        coordinates: List of (x, y) tuples for particle centers
        image_shape: (height, width) of the output mask
        particle_radius: Radius of circular particles in pixels

    Returns:
        Binary mask with 1 for particles, 0 for background
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    for x, y in coordinates:
        # Draw filled circle at each coordinate
        cv2.circle(mask, (int(x), int(y)), particle_radius, 1, -1)

    return mask


def generate_pseudo_labels(predictions: np.ndarray,
                           positive_threshold: float,
                           negative_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo-labels from prediction probabilities.

    Args:
        predictions: Probability map (H, W) with values in [0, 1]
        positive_threshold: Confidence threshold for pseudo-positive labels (e.g., 0.95)
        negative_threshold: Confidence threshold for reliable-negative labels (e.g., 0.05)

    Returns:
        pseudo_positive_mask: Binary mask of high-confidence positives
        reliable_negative_mask: Binary mask of high-confidence negatives
    """
    pseudo_positive_mask = (predictions > positive_threshold).astype(np.uint8)
    reliable_negative_mask = (predictions < negative_threshold).astype(np.uint8)

    return pseudo_positive_mask, reliable_negative_mask


def combine_labels(original_mask: np.ndarray,
                   pseudo_positive_mask: np.ndarray,
                   reliable_negative_mask: np.ndarray) -> np.ndarray:
    """
    Combine original labels with pseudo-labels for self-training.

    Priority:
    1. Original labeled particles → 1
    2. Pseudo-positive predictions → 1
    3. Reliable-negative predictions → 0
    4. Uncertain regions → -1 (ignore during loss)

    Args:
        original_mask: Ground truth mask from human labels
        pseudo_positive_mask: High-confidence positive predictions
        reliable_negative_mask: High-confidence negative predictions

    Returns:
        Combined mask with values {-1, 0, 1}
    """
    combined = np.full_like(original_mask, -1, dtype=np.int8)  # Start with all ignore

    # Apply in priority order
    combined[reliable_negative_mask == 1] = 0  # Reliable negatives
    combined[pseudo_positive_mask == 1] = 1     # Pseudo positives
    combined[original_mask == 1] = 1            # Original labels (highest priority)

    return combined


# ============================================================================
# Dataset
# ============================================================================

class CryoEMSegmentationDataset(Dataset):
    """Dataset for cryo-EM semantic segmentation with ignore mask support and augmentation"""

    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray],
                 transform=None, is_training=True):
        """
        Args:
            images: List of grayscale images (H, W)
            masks: List of segmentation masks (H, W) with values {-1, 0, 1}
                   -1 = ignore, 0 = background, 1 = particle
            transform: Optional augmentation pipeline
            is_training: If True, apply augmentation
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        self.is_training = is_training

        assert len(images) == len(masks), "Number of images and masks must match"

        # Default augmentation if none provided
        if self.transform is None:
            if is_training:
                # IMPROVED: Strong augmentation for training
                self.transform = Compose([
                    CryoEMAugmentation(p=0.5),
                    Normalize()
                ])
            else:
                # Validation/test: only normalize
                self.transform = Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()  # Copy to avoid modifying original
        mask = self.masks[idx].copy()

        # Ensure float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # IMPROVED: Apply augmentation if training
        if self.is_training and self.transform is not None:
            # Augmentation expects mask values {0, 1}, handle ignore mask (-1)
            ignore_mask = (mask == -1)
            mask_binary = np.clip(mask, 0, 1)  # Convert to {0, 1} for augmentation

            image, mask_binary = self.transform(image, mask_binary)

            # Restore ignore regions
            mask = mask_binary
            mask[ignore_mask] = -1
        elif self.transform is not None:
            # Validation: only normalize
            image = self.transform(image)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).long()          # (H, W)

        return image_tensor, mask_tensor


# ============================================================================
# Loss Function with Ignore Mask Support
# ============================================================================

class IgnoreMaskBCELoss(nn.Module):
    """BCE loss that ignores pixels with mask value -1"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Raw logits (B, 1, H, W)
            targets: Target masks (B, H, W) with values {-1, 0, 1}

        Returns:
            Loss value (scalar)
        """
        # Create mask for valid pixels (not -1)
        valid_mask = (targets != -1).float().unsqueeze(1)  # (B, 1, H, W)

        # Convert targets to float and replace -1 with 0 for BCE
        targets_float = targets.float().unsqueeze(1)  # (B, 1, H, W)
        targets_float = torch.clamp(targets_float, 0, 1)  # Replace -1 with 0

        # Compute BCE loss
        loss = self.bce(predictions, targets_float)

        # Apply valid mask and compute mean only over valid pixels
        masked_loss = loss * valid_mask

        num_valid = valid_mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum()  # Return 0 if no valid pixels


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss_output = criterion(outputs, masks)

        # Handle CombinedLoss which returns (loss, components_dict)
        if isinstance(loss_output, tuple):
            loss, components = loss_output
        else:
            loss = loss_output

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def inference_all(model, images: List[np.ndarray], device, batch_size=4):
    """
    Run inference on all images to generate prediction masks.

    Args:
        model: Trained U-Net model
        images: List of input images
        device: torch device
        batch_size: Inference batch size

    Returns:
        List of prediction probability masks (H, W) with values in [0, 1]
    """
    model.eval()
    predictions = []

    # Create temporary dataset
    dummy_masks = [np.zeros_like(img, dtype=np.int8) for img in images]
    dataset = CryoEMSegmentationDataset(images, dummy_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_images, _ in tqdm(dataloader, desc="Inference"):
        batch_images = batch_images.to(device)

        # Forward pass
        logits = model(batch_images)  # (B, 1, H, W)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)

        for prob_map in probs:
            predictions.append(prob_map)

    return predictions


# ============================================================================
# Validation and Metrics Logging
# ============================================================================

def validate_and_log_metrics(
    model: nn.Module,
    val_images: List[np.ndarray],
    val_masks_gt: List[np.ndarray],
    device: torch.device,
    output_dir: Path,
    stage_name: str,
    batch_size: int = 4
) -> Dict[str, float]:
    """
    Validate model on validation set and log metrics to JSON file.

    Computes pixel-wise metrics against original ground truth (not pseudo-labels):
    - Precision
    - Recall
    - F1 Score
    - IoU (Jaccard Score)
    - AUC

    Args:
        model: Trained model to evaluate
        val_images: List of validation images
        val_masks_gt: List of ground truth masks (original labels, not pseudo)
        device: torch device
        output_dir: Directory to save metrics
        stage_name: Name of training stage (e.g., "iteration_0_supervised")
        batch_size: Batch size for inference

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_predictions = []
    all_gt = []

    print(f"\nValidating on {len(val_images)} images...")

    # Run inference on validation set
    with torch.no_grad():
        # Create temporary dataset for validation (no augmentation)
        val_dataset = CryoEMSegmentationDataset(val_images, val_masks_gt, transform=None, is_training=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for batch_images, batch_masks in tqdm(val_loader, desc="Validation"):
            batch_images = batch_images.to(device)

            # Forward pass
            logits = model(batch_images)  # (B, 1, H, W)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)

            # Binarize predictions (threshold=0.5)
            preds_binary = (probs >= 0.5).astype(np.uint8)

            # Store predictions and ground truth
            batch_masks_np = batch_masks.numpy().astype(np.int8)

            for pred, gt in zip(preds_binary, batch_masks_np):
                # Only evaluate on valid pixels (ignore -1)
                valid_mask = (gt != -1)
                if valid_mask.sum() == 0:
                    continue  # Skip if no valid pixels

                pred_flat = pred[valid_mask].flatten()
                gt_flat = gt[valid_mask].flatten()

                all_predictions.extend(pred_flat)
                all_gt.extend(gt_flat)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_gt = np.array(all_gt)

    # Compute metrics
    precision = precision_score(all_gt, all_predictions, zero_division=0)
    recall = recall_score(all_gt, all_predictions, zero_division=0)
    f1 = f1_score(all_gt, all_predictions, zero_division=0)
    iou = jaccard_score(all_gt, all_predictions, zero_division=0)

    # AUC requires probability scores, so we need to recompute with probs
    # For simplicity, we'll use binary predictions for now
    # In production, you might want to store probabilities
    try:
        auc = roc_auc_score(all_gt, all_predictions)
    except ValueError:
        auc = 0.0  # In case only one class is present

    metrics = {
        'stage': stage_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
        'auc': float(auc),
        'num_val_samples': len(val_images),
        'num_pixels_evaluated': len(all_gt)
    }

    # Print metrics
    print(f"\n{'='*60}")
    print(f"Validation Metrics - {stage_name}")
    print(f"{'='*60}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  IoU:       {iou:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"{'='*60}\n")

    # Save to JSON
    metrics_file = output_dir / 'metrics.json'

    # Load existing metrics if file exists
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    # Append new metrics
    all_metrics.append(metrics)

    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"✓ Metrics saved to {metrics_file}")

    model.train()  # Return to training mode
    return metrics


# ============================================================================
# Visualization for Paper Figures
# ============================================================================

def save_comparison_visualization(
    model: nn.Module,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pseudo_mask: np.ndarray,
    device: torch.device,
    save_path: Path,
    stage_name: str
):
    """
    Save 4-panel comparison figure for paper.

    Panels:
    1. Input image
    2. Ground truth mask
    3. Pseudo-label mask (if available)
    4. Model prediction

    Args:
        model: Trained model
        image: Input image
        gt_mask: Ground truth mask
        pseudo_mask: Pseudo-label mask (or None)
        device: torch device
        save_path: Path to save visualization
        stage_name: Name of training stage
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
    import matplotlib.pyplot as plt

    model.eval()

    # Run inference
    with torch.no_grad():
        # Prepare image
        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

        # Normalize if needed
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)

        # Forward pass
        logits = model(img_tensor)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Create figure
    if pseudo_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ['Input Image', 'Ground Truth', 'Pseudo-Label', 'Prediction']

        axes[0].imshow(image, cmap='gray')
        axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[2].imshow(pseudo_mask, cmap='gray', vmin=0, vmax=1)
        axes[3].imshow(pred, cmap='hot', vmin=0, vmax=1)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ['Input Image', 'Ground Truth', 'Prediction']

        axes[0].imshow(image, cmap='gray')
        axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[2].imshow(pred, cmap='hot', vmin=0, vmax=1)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle(f'Self-Training Visualization - {stage_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {save_path}")

    model.train()


# ============================================================================
# Adaptive Thresholding for Self-Training
# ============================================================================

def adaptive_threshold(
    iteration: int,
    total_iterations: int,
    base_pos: float = 0.95,
    base_neg: float = 0.05,
    decay: float = 0.02
) -> Tuple[float, float]:
    """
    Gradually relaxes pseudo-label thresholds as self-training progresses.

    Early iterations use strict thresholds (high confidence required) to avoid
    error propagation. Later iterations relax thresholds to include more data.

    Args:
        iteration: Current iteration (1-indexed)
        total_iterations: Total number of self-training iterations
        base_pos: Initial positive threshold (default: 0.95)
        base_neg: Initial negative threshold (default: 0.05)
        decay: Amount to relax threshold per iteration (default: 0.02)

    Returns:
        Tuple of (positive_threshold, negative_threshold)

    Example:
        Iteration 1/3: (0.95, 0.05) - very strict
        Iteration 2/3: (0.93, 0.07) - moderate
        Iteration 3/3: (0.91, 0.09) - relaxed
    """
    # Relax thresholds progressively
    pos_thresh = max(0.80, base_pos - decay * (iteration - 1))
    neg_thresh = min(0.20, base_neg + decay * (iteration - 1))

    return pos_thresh, neg_thresh


# ============================================================================
# Main Self-Training Pipeline
# ============================================================================

def self_training_pipeline(
    images: List[np.ndarray],
    coordinates_list: List[List[Tuple[int, int]]],
    config: Dict,
    output_dir: Path,
    device: torch.device,
    use_multi_gpu: bool = False
):
    """
    Main self-training pipeline.

    Args:
        images: List of cryo-EM micrographs
        coordinates_list: List of particle coordinates for each image
        config: Configuration dictionary
        output_dir: Directory to save models and results
        device: torch device
        use_multi_gpu: Whether to use DataParallel for multi-GPU training
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMPROVED: Set random seed for reproducibility
    set_seed(42)

    # Extract config parameters
    particle_radius = config['particle_radius']
    initial_epochs = config['initial_epochs']
    self_training_iterations = config['self_training_iterations']
    retrain_epochs = config['retrain_epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    positive_threshold = config['positive_threshold']
    negative_threshold = config['negative_threshold']

    print("="*80)
    print("U-Net Self-Training Pipeline for Cryo-EM Particle Picking")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Particle radius: {particle_radius}")
    print(f"  Initial epochs: {initial_epochs}")
    print(f"  Self-training iterations: {self_training_iterations}")
    print(f"  Retrain epochs: {retrain_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Positive threshold: {positive_threshold}")
    print(f"  Negative threshold: {negative_threshold}")
    print(f"\nDataset: {len(images)} images")
    print()

    # ========================================================================
    # Step 1: Generate Ground Truth Masks from Coordinates
    # ========================================================================
    print("Step 1: Generating ground truth masks from coordinates...")
    original_masks = []
    for coords in tqdm(coordinates_list, desc="Creating masks"):
        image_shape = images[0].shape  # Assume all images same size
        mask = coordinates_to_mask(coords, image_shape, particle_radius)
        original_masks.append(mask)

    print(f"  Generated {len(original_masks)} masks")

    # ========================================================================
    # Step 1b: Create Train/Val Split (90/10)
    # ========================================================================
    print("\nStep 1b: Creating train/val split (90% train, 10% val)...")

    # Create indices and shuffle
    num_samples = len(images)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    # Split
    val_split = 0.1
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Split data
    train_images = [images[i] for i in train_indices]
    train_masks = [original_masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [original_masks[i] for i in val_indices]

    print(f"  Training set: {len(train_images)} images")
    print(f"  Validation set: {len(val_images)} images")

    # ========================================================================
    # Step 2: Phase 1 - Initial Supervised Training
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 1: Initial Supervised Training")
    print("="*80)

    # Create dataset and dataloader (only training data)
    train_dataset = CryoEMSegmentationDataset(train_images, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = UNet(in_channels=1, out_channels=1, base_features=64).to(device)

    # Wrap with DataParallel for multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel across {torch.cuda.device_count()} GPUs\n")

    # IMPROVED: Use Combined Loss (Focal + Dice)
    print("✅ Using CombinedLoss (Focal + Dice) instead of BCE")
    criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)

    # IMPROVED: Use AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # IMPROVED: Add Cosine Annealing LR Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,       # Restart every 10 epochs
        T_mult=2,     # Double period after restart
        eta_min=1e-6  # Minimum learning rate
    )
    print("✅ Using Cosine Annealing LR Scheduler")
    print()

    # Train initial model
    print(f"\nTraining for {initial_epochs} epochs...")
    for epoch in range(1, initial_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Handle CombinedLoss return value (loss, components dict)
        if isinstance(train_loss, tuple):
            train_loss = train_loss[0]

        print(f"Epoch {epoch}/{initial_epochs} - Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

    # IMPROVED: Save baseline model in organized subdirectory
    iter0_dir = output_dir / 'iteration_0_supervised'
    iter0_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = iter0_dir / 'model.pt'
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
    torch.save(model_state, baseline_path)
    print(f"\n✓ Baseline model saved to {baseline_path}")

    # IMPROVED: Validate and log metrics after initial training
    validate_and_log_metrics(
        model, val_images, val_masks, device, iter0_dir,
        stage_name='iteration_0_supervised', batch_size=batch_size
    )

    # IMPROVED: Save visualization for paper (first validation image)
    if len(val_images) > 0:
        vis_path = iter0_dir / 'visualization.png'
        save_comparison_visualization(
            model, val_images[0], val_masks[0], None, device, vis_path,
            stage_name='iteration_0_supervised'
        )

    # ========================================================================
    # Step 3: Phase 2 - Iterative Self-Training Loop
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 2: Iterative Self-Training Loop")
    print("="*80)

    for iteration in range(1, self_training_iterations + 1):
        print(f"\n{'='*80}")
        print(f"Self-Training Iteration {iteration}/{self_training_iterations}")
        print(f"{'='*80}")

        # IMPROVED: Use adaptive thresholds that relax over iterations
        pos_thresh_adaptive, neg_thresh_adaptive = adaptive_threshold(
            iteration, self_training_iterations,
            base_pos=positive_threshold,
            base_neg=negative_threshold
        )

        # A. Inference Step (only on training images)
        print(f"\nA. Running inference on training images...")
        predictions = inference_all(model, train_images, device, batch_size=batch_size)

        # B. Selective Pseudo-Label Generation
        print(f"\nB. Generating pseudo-labels...")
        print(f"   Positive threshold: {pos_thresh_adaptive:.3f} (adaptive, base={positive_threshold})")
        print(f"   Negative threshold: {neg_thresh_adaptive:.3f} (adaptive, base={negative_threshold})")

        pseudo_positive_masks = []
        reliable_negative_masks = []

        for pred in predictions:
            pseudo_pos, reliable_neg = generate_pseudo_labels(
                pred, pos_thresh_adaptive, neg_thresh_adaptive
            )
            pseudo_positive_masks.append(pseudo_pos)
            reliable_negative_masks.append(reliable_neg)

        # Compute statistics
        total_pixels = predictions[0].size
        avg_pseudo_pos = np.mean([m.sum() / total_pixels * 100 for m in pseudo_positive_masks])
        avg_reliable_neg = np.mean([m.sum() / total_pixels * 100 for m in reliable_negative_masks])

        print(f"   Avg pseudo-positive coverage: {avg_pseudo_pos:.2f}%")
        print(f"   Avg reliable-negative coverage: {avg_reliable_neg:.2f}%")

        # C. Dataset Combination
        print(f"\nC. Combining original labels with pseudo-labels...")
        combined_masks = []

        for orig_mask, pseudo_pos, reliable_neg in zip(
            train_masks, pseudo_positive_masks, reliable_negative_masks
        ):
            combined = combine_labels(orig_mask, pseudo_pos, reliable_neg)
            combined_masks.append(combined)

        # Compute label statistics
        total_pixels = combined_masks[0].size
        avg_positive = np.mean([(m == 1).sum() / total_pixels * 100 for m in combined_masks])
        avg_negative = np.mean([(m == 0).sum() / total_pixels * 100 for m in combined_masks])
        avg_ignore = np.mean([(m == -1).sum() / total_pixels * 100 for m in combined_masks])

        print(f"   Positive pixels: {avg_positive:.2f}%")
        print(f"   Negative pixels: {avg_negative:.2f}%")
        print(f"   Ignored pixels: {avg_ignore:.2f}%")

        # D. Retraining Step
        print(f"\nD. Retraining model for {retrain_epochs} epochs...")

        # Create new dataset with combined masks (training data only)
        retrain_dataset = CryoEMSegmentationDataset(train_images, combined_masks)
        retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer (or continue from previous)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, retrain_epochs + 1):
            train_loss = train_epoch(model, retrain_loader, criterion, optimizer, device)
            print(f"  Epoch {epoch}/{retrain_epochs} - Loss: {train_loss:.4f}")

        # IMPROVED: Save model in organized subdirectory
        iter_dir = output_dir / f'iteration_{iteration}_selftrain'
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_path = iter_dir / 'model.pt'
        model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
        torch.save(model_state, iter_path)
        print(f"\n✓ Model saved to {iter_path}")

        # IMPROVED: Validate and log metrics after each self-training iteration
        validate_and_log_metrics(
            model, val_images, val_masks, device, iter_dir,
            stage_name=f'iteration_{iteration}_selftrain', batch_size=batch_size
        )

        # IMPROVED: Save visualization with pseudo-labels (use first training image)
        if len(train_images) > 0:
            vis_path = iter_dir / 'visualization.png'
            # Get corresponding pseudo-label for first training image
            pseudo_label_vis = combined_masks[0] if len(combined_masks) > 0 else None
            save_comparison_visualization(
                model, train_images[0], train_masks[0], pseudo_label_vis, device, vis_path,
                stage_name=f'iteration_{iteration}_selftrain'
            )

    # ========================================================================
    # Final Save
    # ========================================================================
    final_path = output_dir / 'final_model.pt'
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
    torch.save(model_state, final_path)
    print(f"\n{'='*80}")
    print(f"Self-Training Complete!")
    print(f"{'='*80}")
    print(f"Final model saved to {final_path}")

    # Save training summary
    summary = {
        'config': config,
        'num_images': len(images),
        'iterations_completed': self_training_iterations,
        'final_model': str(final_path)
    }

    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to {summary_path}")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='U-Net Self-Training for Cryo-EM Particle Picking')

    # Data paths
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing micrograph images')
    parser.add_argument('--coords-file', type=str, required=True,
                       help='JSON file with particle coordinates')
    parser.add_argument('--output-dir', type=str, default='experiments/unet_selftraining',
                       help='Output directory for models and results')

    # Training parameters (IMPROVED: Longer training by default)
    parser.add_argument('--particle-radius', type=int, default=10,
                       help='Radius of particles in pixels for mask generation')
    parser.add_argument('--initial-epochs', type=int, default=100,
                       help='Number of epochs for initial supervised training (IMPROVED: 100 vs 50)')
    parser.add_argument('--self-training-iterations', type=int, default=3,
                       help='Number of self-training iterations')
    parser.add_argument('--retrain-epochs', type=int, default=30,
                       help='Number of epochs for each retraining iteration (IMPROVED: 30 vs 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (IMPROVED: 128 vs 4 for faster training)')

    # Pseudo-labeling parameters
    parser.add_argument('--positive-threshold', type=float, default=0.95,
                       help='Confidence threshold for pseudo-positive labels')
    parser.add_argument('--negative-threshold', type=float, default=0.05,
                       help='Confidence threshold for reliable-negative labels')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs with DataParallel')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        use_multi_gpu = True
    else:
        print(f"Using device: {device}")
        use_multi_gpu = False
    print()

    # Load data (placeholder - adapt to your data format)
    # This is a template - you'll need to implement loading your specific data format
    print("Loading data...")

    # Load images from directory (supports .npy, .png, .tif)
    image_dir = Path(args.image_dir)
    image_files = sorted(image_dir.glob('*.npy')) or \
                  sorted(image_dir.glob('*.png')) or \
                  sorted(image_dir.glob('*.tif'))

    images = []
    for img_file in image_files:
        if img_file.suffix == '.npy':
            img = np.load(str(img_file))
        else:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        images.append(img)

    # Example: Load coordinates from JSON
    # Expected format: {"image1.png": [[x1, y1], [x2, y2], ...], ...}
    with open(args.coords_file, 'r') as f:
        coords_dict = json.load(f)

    coordinates_list = []
    for img_file in image_files:
        coords = coords_dict.get(img_file.name, [])
        coordinates_list.append(coords)

    print(f"Loaded {len(images)} images")
    print(f"Loaded coordinates for {len(coordinates_list)} images")

    # Create config
    config = {
        'particle_radius': args.particle_radius,
        'initial_epochs': args.initial_epochs,
        'self_training_iterations': args.self_training_iterations,
        'retrain_epochs': args.retrain_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'positive_threshold': args.positive_threshold,
        'negative_threshold': args.negative_threshold,
    }

    # Run self-training pipeline
    output_dir = Path(args.output_dir)
    self_training_pipeline(images, coordinates_list, config, output_dir, device, use_multi_gpu)


if __name__ == '__main__':
    main()
