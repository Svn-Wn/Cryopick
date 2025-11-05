#!/usr/bin/env python3
"""
Train Attention U-Net for Fair Comparison with Standard U-Net

This script uses IDENTICAL training methodology as train_standard_unet_fair_comparison.py:
- Direct mask-to-mask supervision (no coordinate conversion)
- Same data loading (no re-splitting)
- Same loss function (Focal + Dice)
- Same hyperparameters
- Only difference: Attention gates in the architecture

This ensures a fair architectural comparison.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path

# Import model and utilities
from models.attention_unet import AttentionUNet
from train_unet_selftraining_improved import (
    load_all_from_dir,
    train_epoch,
    validate_and_log_metrics,
    set_seed,
    CryoEMSegmentationDataset
)
from improved_losses import CombinedLoss


def main():
    parser = argparse.ArgumentParser(description='Train Attention U-Net (Fair Comparison Mode)')

    # Data paths
    parser.add_argument('--train-data-dir', required=True,
                       help='Directory with training data')
    parser.add_argument('--val-data-dir', required=True,
                       help='Directory with validation data')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for models and results')

    # Training parameters
    parser.add_argument('--initial-epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print("="*80)
    print("ATTENTION U-NET TRAINING (FAIR COMPARISON MODE)")
    print("="*80)
    print("\nTraining Attention U-Net on FULL-SIZE 768×768 images")
    print("  - Direct mask-to-mask supervision (same as Standard U-Net)")
    print("  - No coordinate conversion")
    print("  - Fair apples-to-apples comparison")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data (EXACTLY like Standard U-Net)
    print("\nLoading training data...")
    train_images, train_masks = load_all_from_dir(args.train_data_dir)
    print(f"  Loaded {len(train_images)} training images")
    print(f"  Image shape: {train_images[0].shape}")

    print("\nLoading validation data...")
    val_images, val_masks = load_all_from_dir(args.val_data_dir)
    print(f"  Loaded {len(val_images)} validation images")
    print(f"  Image shape: {val_images[0].shape}")

    print(f"\nDataset:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")

    # Create datasets
    train_dataset = CryoEMSegmentationDataset(train_images, train_masks, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    print("\n" + "="*80)
    print("Initializing Attention U-Net")
    print("="*80)

    model = AttentionUNet(in_channels=1, out_channels=1)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Attention U-Net")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: 4 encoder levels + attention gates + bottleneck + 4 decoder levels")

    # Loss and optimizer (SAME as Standard U-Net)
    criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Create output directory
    output_dir = Path(args.output_dir)
    iter0_dir = output_dir / 'iteration_0_supervised'
    iter0_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print("\n" + "="*80)
    print("Phase 1: Supervised Training")
    print("="*80)

    best_f1 = 0.0
    metrics_history = []

    for epoch in range(1, args.initial_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if isinstance(train_loss, tuple):
            train_loss = train_loss[0]

        print(f"Epoch {epoch}/{args.initial_epochs} - Loss: {train_loss:.4f}")

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == args.initial_epochs:
            metrics = validate_and_log_metrics(
                model, val_images, val_masks, device, iter0_dir,
                f'epoch_{epoch}', batch_size=args.batch_size
            )

            # Save best model
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                torch.save(model.state_dict(), iter0_dir / 'best_model.pt')
                print(f"  ✅ New best F1: {best_f1:.4f}")

    # Save final model
    torch.save(model.state_dict(), iter0_dir / 'model.pt')

    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)
    print(f"\nBest F1 Score: {best_f1:.4f}")
    print(f"Results saved to: {iter0_dir}")
    print("\n✅ FAIR COMPARISON: Trained with same methodology as Standard U-Net!")
    print("   Ready for fair comparison.")
    print("="*80)


if __name__ == '__main__':
    main()
