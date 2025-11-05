#!/usr/bin/env python3
"""
Standalone Inference Script for Trained U-Net Model

Usage:
    python inference_standalone.py --model model.pt --image input.png --output output.png
    python inference_standalone.py --model model.pt --image-dir images/ --output-dir predictions/

Requirements:
    pip install torch numpy opencv-python
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm


# ============================================================================
# U-Net Architecture (same as training)
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

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)


# ============================================================================
# Inference Functions
# ============================================================================

def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = UNet(in_channels=1, out_channels=1, base_features=64)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image):
    """Preprocess image for inference"""
    # Convert to float32
    image = image.astype(np.float32)

    # Normalize (z-score normalization)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std

    return image


def predict_single_image(model, image, device='cuda', threshold=0.5):
    """
    Run inference on a single image

    Args:
        model: Trained UNet model
        image: Input image (H, W) numpy array, grayscale
        device: Device to run on ('cuda' or 'cpu')
        threshold: Threshold for binary prediction (default: 0.5)

    Returns:
        prob_map: Probability map (H, W) numpy array, values in [0, 1]
        binary_mask: Binary mask (H, W) numpy array, values in {0, 1}
    """
    # Preprocess
    image_preprocessed = preprocess_image(image)

    # Convert to tensor [1, 1, H, W]
    image_tensor = torch.from_numpy(image_preprocessed).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    # Convert to numpy
    prob_map = probabilities.squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)

    return prob_map, binary_mask


def extract_coordinates(binary_mask, min_area=10):
    """
    Extract particle coordinates from binary mask

    Args:
        binary_mask: Binary mask (H, W) with values {0, 1}
        min_area: Minimum area for a particle (pixels)

    Returns:
        coordinates: List of (x, y) tuples for particle centers
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    coordinates = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[i]
            coordinates.append((int(cx), int(cy)))

    return coordinates


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained U-Net model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image-dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Path to output prediction')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--save-prob', action='store_true', help='Save probability maps')
    parser.add_argument('--save-coords', action='store_true', help='Save particle coordinates')

    args = parser.parse_args()

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    print("✅ Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Single image mode
    if args.image:
        print(f"\nProcessing image: {args.image}")

        # Load image
        image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {args.image}")

        print(f"  Image shape: {image.shape}")

        # Run inference
        prob_map, binary_mask = predict_single_image(model, image, device, args.threshold)

        print(f"  Positive pixels: {binary_mask.sum()} / {binary_mask.size} ({binary_mask.sum() / binary_mask.size * 100:.2f}%)")

        # Save outputs
        if args.output:
            # Save binary mask
            cv2.imwrite(args.output, binary_mask * 255)
            print(f"  ✅ Saved prediction to: {args.output}")

            # Save probability map if requested
            if args.save_prob:
                prob_output = args.output.replace('.png', '_prob.png')
                cv2.imwrite(prob_output, (prob_map * 255).astype(np.uint8))
                print(f"  ✅ Saved probability map to: {prob_output}")

            # Save coordinates if requested
            if args.save_coords:
                coords = extract_coordinates(binary_mask)
                coords_output = args.output.replace('.png', '_coords.txt')
                with open(coords_output, 'w') as f:
                    for x, y in coords:
                        f.write(f"{x}\t{y}\n")
                print(f"  ✅ Saved {len(coords)} coordinates to: {coords_output}")

        return

    # Batch mode
    if args.image_dir:
        print(f"\nProcessing directory: {args.image_dir}")

        image_dir = Path(args.image_dir)
        output_dir = Path(args.output_dir) if args.output_dir else image_dir / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + \
                      list(image_dir.glob('*.tif')) + list(image_dir.glob('*.tiff'))

        print(f"Found {len(image_files)} images")

        # Process each image
        for image_path in tqdm(image_files, desc="Processing"):
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"  ⚠ Could not load: {image_path}")
                continue

            # Run inference
            prob_map, binary_mask = predict_single_image(model, image, device, args.threshold)

            # Save outputs
            output_path = output_dir / f"{image_path.stem}_pred.png"
            cv2.imwrite(str(output_path), binary_mask * 255)

            if args.save_prob:
                prob_output = output_dir / f"{image_path.stem}_prob.png"
                cv2.imwrite(str(prob_output), (prob_map * 255).astype(np.uint8))

            if args.save_coords:
                coords = extract_coordinates(binary_mask)
                coords_output = output_dir / f"{image_path.stem}_coords.txt"
                with open(coords_output, 'w') as f:
                    for x, y in coords:
                        f.write(f"{x}\t{y}\n")

        print(f"\n✅ Processed {len(image_files)} images")
        print(f"✅ Results saved to: {output_dir}")

        return

    print("❌ Please specify either --image or --image-dir")


if __name__ == '__main__':
    main()
