#!/usr/bin/env python3
"""
Train Standard U-Net for Fair Comparison with Attention U-Net

Trains on the SAME DATA as Attention U-Net:
- 768×768 full-size images
- Same number of samples (570 validation images)
- Same training configuration (100 epochs, batch size, etc.)

This enables an apples-to-apples comparison.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import Standard U-Net and utilities
from train_unet_selftraining_improved import (
    UNet,
    CryoEMSegmentationDataset,
    train_epoch,
    validate_and_log_metrics,
    set_seed,
    load_all_from_dir
)
from improved_losses import CombinedLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def main():
    parser = argparse.ArgumentParser(
        description='Train Standard U-Net for Fair Comparison with Attention U-Net'
    )

    # Data paths
    parser.add_argument('--train-data-dir', required=True,
                       help='Directory with training data (batch_*.npz files)')
    parser.add_argument('--val-data-dir', required=True,
                       help='Directory with validation data')
    parser.add_argument('--output-dir', default='experiments/standard_unet_fair',
                       help='Output directory for models and results')

    # Training parameters (match Attention U-Net configuration)
    parser.add_argument('--initial-epochs', type=int, default=100,
                       help='Initial supervised training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0, cpu, etc.)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use DataParallel for multi-GPU')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1

    print("="*80)
    print("STANDARD U-NET TRAINING (FAIR COMPARISON MODE)")
    print("="*80)
    print(f"\nTraining Standard U-Net on FULL-SIZE 768×768 images")
    print(f"  - Same data format as Attention U-Net")
    print(f"  - Fair apples-to-apples comparison")
    print()

    if use_multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print(f"Using device: {device}")
    print()

    # Load data
    print("Loading training data...")
    train_images, train_masks = load_all_from_dir(args.train_data_dir)
    print(f"  Loaded {len(train_images)} training images")
    print(f"  Image shape: {train_images[0].shape}")

    print("\nLoading validation data...")
    val_images, val_masks = load_all_from_dir(args.val_data_dir)
    print(f"  Loaded {len(val_images)} validation images")
    print(f"  Image shape: {val_images[0].shape}")

    # Data is already split into train and val directories, use directly
    train_imgs = train_images
    train_msks = train_masks
    val_imgs = val_images
    val_msks = val_masks

    print(f"\nDataset:")
    print(f"  Training: {len(train_imgs)} images")
    print(f"  Validation: {len(val_imgs)} images")

    # Create datasets
    train_dataset = CryoEMSegmentationDataset(train_imgs, train_msks, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize Standard U-Net
    print("\n" + "="*80)
    print("Initializing Standard U-Net")
    print("="*80)

    model = UNet(in_channels=1, out_channels=1, base_features=64).to(device)

    if use_multi_gpu:
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Standard U-Net")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: 4 encoder levels + bottleneck + 4 decoder levels")

    # Loss and optimizer (same as Attention U-Net)
    criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Phase 1: Initial training
    print("\n" + "="*80)
    print("Phase 1: Supervised Training")
    print("="*80)

    output_dir = Path(args.output_dir)
    iter0_dir = output_dir / 'iteration_0_supervised'
    iter0_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0

    for epoch in range(1, args.initial_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if isinstance(train_loss, tuple):
            train_loss = train_loss[0]

        print(f"Epoch {epoch}/{args.initial_epochs} - Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.initial_epochs:
            metrics = validate_and_log_metrics(
                model, val_imgs, val_msks, device, iter0_dir,
                f'epoch_{epoch}', batch_size=args.batch_size
            )
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
                torch.save(model_state, iter0_dir / 'best_model.pt')
                print(f"  ✅ New best F1: {best_f1:.4f}")

    # Save final
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
    torch.save(model_state, iter0_dir / 'model.pt')

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest F1 Score: {best_f1:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"\n✅ Standard U-Net trained on same data as Attention U-Net!")
    print(f"   Ready for fair comparison.")


if __name__ == '__main__':
    main()
