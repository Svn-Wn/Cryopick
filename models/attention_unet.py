#!/usr/bin/env python3
"""
Attention U-Net for CryoEM Particle Picking

Attention U-Net adds attention gates to the standard U-Net architecture,
allowing the model to focus on salient features and suppress irrelevant
background regions. This is particularly useful for low SNR CryoEM images.

Key Features:
- Attention gates before each skip connection
- Learns to focus on particle regions automatically
- Better handling of noisy backgrounds
- Improved feature fusion in decoder

References:
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- Schlemper et al., "Attention gated networks: Learning to leverage salient
  regions in medical images" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate Module

    Learns to focus on salient features by computing attention coefficients
    that weight the skip connection features.

    Args:
        F_g: Number of feature channels from gating signal (decoder)
        F_l: Number of feature channels from skip connection (encoder)
        F_int: Number of intermediate feature channels
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Transform gating signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)

        Returns:
            x_att: Attention-weighted features (B, F_l, H, W)
        """
        # Ensure spatial dimensions match
        if g.shape[2:] != x.shape[2:]:
            # Upsample gating signal to match skip connection size
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Transform both inputs
        g1 = self.W_g(g)  # (B, F_int, H, W)
        x1 = self.W_x(x)  # (B, F_int, H, W)

        # Element-wise addition + ReLU
        psi_input = self.relu(g1 + x1)  # (B, F_int, H, W)

        # Compute attention coefficients
        attention = self.psi(psi_input)  # (B, 1, H, W)

        # Apply attention to skip connection
        x_att = x * attention  # (B, F_l, H, W)

        return x_att


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


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Semantic Segmentation

    Architecture:
    - Standard U-Net encoder-decoder structure
    - Attention gates applied to skip connections
    - Learns to focus on relevant features automatically

    Particularly effective for:
    - Low signal-to-noise ratio images (like CryoEM)
    - Images with complex backgrounds
    - Tasks requiring precise localization

    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for binary segmentation)
        base_features: Number of features in first layer (default: 64)
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super(AttentionUNet, self).__init__()

        features = base_features

        # ====================================================================
        # Encoder (Contracting Path)
        # ====================================================================
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ====================================================================
        # Bottleneck
        # ====================================================================
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # ====================================================================
        # Attention Gates (one for each skip connection)
        # ====================================================================
        # Attention gate 4: Between upsampled bottleneck and enc4
        # After upconv4, we have features*8 channels
        self.att4 = AttentionGate(
            F_g=features * 8,   # From upconv4 (upsampled bottleneck)
            F_l=features * 8,   # From encoder (enc4)
            F_int=features * 4  # Intermediate features
        )

        # Attention gate 3: Between upsampled dec4 and enc3
        # After upconv3, we have features*4 channels
        self.att3 = AttentionGate(
            F_g=features * 4,   # From upconv3
            F_l=features * 4,   # From encoder (enc3)
            F_int=features * 2
        )

        # Attention gate 2: Between upsampled dec3 and enc2
        # After upconv2, we have features*2 channels
        self.att2 = AttentionGate(
            F_g=features * 2,   # From upconv2
            F_l=features * 2,   # From encoder (enc2)
            F_int=features
        )

        # Attention gate 1: Between upsampled dec2 and enc1
        # After upconv1, we have features channels
        self.att1 = AttentionGate(
            F_g=features,       # From upconv1
            F_l=features,       # From encoder (enc1)
            F_int=features // 2
        )

        # ====================================================================
        # Decoder (Expanding Path)
        # ====================================================================
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(features * 16, features * 8)  # 16 = 8 + 8 (skip)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(features * 8, features * 4)  # 8 = 4 + 4 (skip)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(features * 4, features * 2)  # 4 = 2 + 2 (skip)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(features * 2, features)  # 2 = 1 + 1 (skip)

        # ====================================================================
        # Output Layer
        # ====================================================================
        self.out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through Attention U-Net

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            logits: Output logits (B, 1, H, W)
        """
        # ====================================================================
        # Encoder
        # ====================================================================
        enc1 = self.enc1(x)         # (B, 64, H, W)

        enc2 = self.enc2(self.pool1(enc1))  # (B, 128, H/2, W/2)

        enc3 = self.enc3(self.pool2(enc2))  # (B, 256, H/4, W/4)

        enc4 = self.enc4(self.pool3(enc3))  # (B, 512, H/8, W/8)

        # ====================================================================
        # Bottleneck
        # ====================================================================
        bottleneck = self.bottleneck(self.pool4(enc4))  # (B, 1024, H/16, W/16)

        # ====================================================================
        # Decoder with Attention Gates
        # ====================================================================

        # Decoder Level 4
        dec4 = self.upconv4(bottleneck)  # (B, 512, H/8, W/8)
        enc4_att = self.att4(g=dec4, x=enc4)  # Apply attention to skip connection
        dec4 = torch.cat([dec4, enc4_att], dim=1)  # Concatenate
        dec4 = self.dec4(dec4)  # (B, 512, H/8, W/8)

        # Decoder Level 3
        dec3 = self.upconv3(dec4)  # (B, 256, H/4, W/4)
        enc3_att = self.att3(g=dec3, x=enc3)  # Apply attention
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)  # (B, 256, H/4, W/4)

        # Decoder Level 2
        dec2 = self.upconv2(dec3)  # (B, 128, H/2, W/2)
        enc2_att = self.att2(g=dec2, x=enc2)  # Apply attention
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)  # (B, 128, H/2, W/2)

        # Decoder Level 1
        dec1 = self.upconv1(dec2)  # (B, 64, H, W)
        enc1_att = self.att1(g=dec1, x=enc1)  # Apply attention
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)  # (B, 64, H, W)

        # ====================================================================
        # Output
        # ====================================================================
        logits = self.out(dec1)  # (B, 1, H, W)

        return logits

    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization

        Useful for understanding what the model focuses on.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary of attention maps at each level
        """
        # Forward pass through encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Forward pass through decoder and extract attention maps
        attention_maps = {}

        # Level 4
        dec4 = self.upconv4(bottleneck)
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.att4.W_g(dec4)
        x1 = self.att4.W_x(enc4)
        psi_input = self.att4.relu(g1 + x1)
        attention_maps['att4'] = self.att4.psi(psi_input)

        enc4_att = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)

        # Level 3
        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.att3.W_g(dec3)
        x1 = self.att3.W_x(enc3)
        psi_input = self.att3.relu(g1 + x1)
        attention_maps['att3'] = self.att3.psi(psi_input)

        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)

        # Level 2
        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.att2.W_g(dec2)
        x1 = self.att2.W_x(enc2)
        psi_input = self.att2.relu(g1 + x1)
        attention_maps['att2'] = self.att2.psi(psi_input)

        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)

        # Level 1
        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.att1.W_g(dec1)
        x1 = self.att1.W_x(enc1)
        psi_input = self.att1.relu(g1 + x1)
        attention_maps['att1'] = self.att1.psi(psi_input)

        return attention_maps


def test_attention_unet():
    """Test Attention U-Net with sample input"""
    print("Testing Attention U-Net...")

    # Create model
    model = AttentionUNet(in_channels=1, out_channels=1, base_features=64)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 128)

    print(f"\nInput shape: {input_tensor.shape}")

    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # Test attention map extraction
    attention_maps = model.get_attention_maps(input_tensor)
    print(f"\nAttention maps:")
    for name, att_map in attention_maps.items():
        print(f"  {name}: {att_map.shape}")

    print("\n✅ Attention U-Net test passed!")


if __name__ == '__main__':
    test_attention_unet()
