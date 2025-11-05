# Model Architecture Confirmation

**Date**: October 17, 2025
**Status**: ✅ CONFIRMED - Using U-Net for Class-Balanced Pseudo-Labeling

---

## Yes, We're Using U-Net!

### Model Specifications:

```
Architecture:        U-Net (Standard implementation)
Input channels:      1 (grayscale images)
Output channels:     1 (binary segmentation - particle vs background)
Base features:       64
Total parameters:    31,042,369 (~31M parameters)
Trainable params:    31,042,369 (all parameters are trainable)
Image size:          768 × 768 pixels
```

---

## U-Net Architecture Details

### Encoder (Downsampling Path):

```
Input: 1 × 768 × 768

enc1:  1 → 64   | Conv → BN → ReLU → Conv → BN → ReLU
pool1: MaxPool2d (2×2)
       ↓ 64 × 384 × 384

enc2:  64 → 128 | Conv → BN → ReLU → Conv → BN → ReLU
pool2: MaxPool2d (2×2)
       ↓ 128 × 192 × 192

enc3:  128 → 256 | Conv → BN → ReLU → Conv → BN → ReLU
pool3: MaxPool2d (2×2)
       ↓ 256 × 96 × 96

enc4:  256 → 512 | Conv → BN → ReLU → Conv → BN → ReLU
pool4: MaxPool2d (2×2)
       ↓ 512 × 48 × 48
```

### Bottleneck:

```
bottleneck: 512 → 1024 | Conv → BN → ReLU → Conv → BN → ReLU
            ↓ 1024 × 48 × 48
```

### Decoder (Upsampling Path with Skip Connections):

```
upconv4: 1024 → 512 | ConvTranspose2d (2×2)
dec4:    1024 → 512 | Conv → BN → ReLU → Conv → BN → ReLU (concat with enc4)
         ↓ 512 × 96 × 96

upconv3: 512 → 256 | ConvTranspose2d (2×2)
dec3:    512 → 256 | Conv → BN → ReLU → Conv → BN → ReLU (concat with enc3)
         ↓ 256 × 192 × 192

upconv2: 256 → 128 | ConvTranspose2d (2×2)
dec2:    256 → 128 | Conv → BN → ReLU → Conv → BN → ReLU (concat with enc2)
         ↓ 128 × 384 × 384

upconv1: 128 → 64 | ConvTranspose2d (2×2)
dec1:    128 → 64 | Conv → BN → ReLU → Conv → BN → ReLU (concat with enc1)
         ↓ 64 × 768 × 768
```

### Output Layer:

```
out: 64 → 1 | Conv2d (1×1)
     ↓ 1 × 768 × 768

Final activation: Sigmoid (applied during inference)
```

---

## How U-Net is Used in Pseudo-Labeling

### Step 1: Supervised Baseline Training

```python
# Create fresh U-Net model
model_supervised = UNet(in_channels=1, out_channels=1, base_features=64)

# Train on labeled data (258 images at 5%)
# Loss: CombinedLoss (70% Focal + 30% Dice, pos_weight=5.0)
# Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
# Scheduler: CosineAnnealingLR
# Epochs: 100

→ Result: 70% F1 score
```

### Step 2: Generate Class-Balanced Pseudo-Labels

```python
# Use trained model to predict on unlabeled data
model_supervised.eval()

with torch.no_grad():
    for unlabeled_image in unlabeled_images:
        # Forward pass
        logits = model_supervised(unlabeled_image)
        probs = torch.sigmoid(logits)

        # Apply class-specific confidence thresholds
        particle_pixels = probs >= 0.5
        background_pixels = probs < 0.5

        particle_conf = probs[particle_pixels]
        background_conf = 1 - probs[background_pixels]

        # Keep if:
        # 1. Particle confidence >= 0.60
        # 2. Background confidence >= 0.80
        # 3. Particle ratio close to 4%

        if meets_criteria:
            pseudo_mask = (probs >= 0.5)
            add_to_pseudo_labels(unlabeled_image, pseudo_mask)

→ Result: ~500 high-quality pseudo-labels (not 3,000!)
```

### Step 3: Train New U-Net on Combined Data

```python
# Create ANOTHER fresh U-Net model (not reusing the supervised one)
model_iteration1 = UNet(in_channels=1, out_channels=1, base_features=64)

# Train on labeled (258) + pseudo-labeled (~500) = ~758 images
# Same loss, optimizer, scheduler as Step 1
# Epochs: 50 (fewer for iterations)

→ Expected: 70% → 73% F1 score (+3%)
```

### Step 4: Iterate (if improved)

```python
# If iteration 1 improved F1:
# Use model_iteration1 to generate new pseudo-labels
# Train model_iteration2 on labeled + new_pseudo_labels
# Repeat up to 3 times
```

---

## Loss Function

Using **CombinedLoss** (custom implementation):

```python
CombinedLoss(
    focal_weight=0.7,   # 70% Focal Loss
    dice_weight=0.3,    # 30% Dice Loss
    pos_weight=5.0      # 5× weight for particle class
)

# Focal Loss: Handles class imbalance by down-weighting easy examples
# Dice Loss: Directly optimizes F1-like metric
# pos_weight=5.0: Compensates for 96% background, 4% particles
```

---

## Training Configuration

### Optimizer:
```python
AdamW(
    lr=1e-4,
    weight_decay=1e-4  # L2 regularization
)
```

### Scheduler:
```python
CosineAnnealingLR(
    T_max=epochs  # Cosine decay over training
)
```

### Data Augmentation:
```python
# During training (is_training=True):
- Random horizontal/vertical flips
- Random rotations
- Normalize to [0, 1]

# During inference (is_training=False):
- Only normalize to [0, 1]
```

---

## Comparison: U-Net vs Other Architectures

| Architecture | Parameters | Typical Use | Our Choice |
|--------------|------------|-------------|------------|
| **U-Net (Ours)** | **31M** | **Medical imaging, segmentation** | **✅ Using** |
| ResNet-50 | 25M | Classification | ❌ Not suitable for segmentation |
| Transformer | 80-300M | Various tasks | ❌ Too complex, slow |
| SegNet | 29M | Segmentation | ❌ U-Net is better |
| DeepLab | 40M+ | Segmentation | ❌ Overkill for this task |

**Why U-Net?**
1. ✅ Proven effective for biomedical image segmentation
2. ✅ Skip connections preserve fine-grained spatial information
3. ✅ Reasonable parameter count (~31M)
4. ✅ Fast training and inference
5. ✅ Works well with limited data

---

## Model Instantiation in Code

### Supervised Baseline:
```python
# train_pseudolabel_coco_balanced.py, line 411
model_sup = UNet(in_channels=1, out_channels=1, base_features=64).to(device)
```

### Pseudo-Labeling Iterations:
```python
# train_pseudolabel_coco_balanced.py, line 469
new_model = UNet(in_channels=1, out_channels=1, base_features=64).to(device)
```

**Important**: Each iteration creates a **fresh** U-Net model (not fine-tuning the previous one).

---

## Is This the Same U-Net Across All Experiments?

### Yes! The same U-Net architecture is used in:

✅ **Supervised baseline** (70% F1 with 5% labels)
✅ **Naive pseudo-labeling** (0% gain - failed)
✅ **Class-balanced pseudo-labeling** (running now - expected +3-5%)
✅ **FixMatch** (catastrophic failure - recall collapsed)
✅ **All previous experiments**

**The ONLY difference is the training data and pseudo-label selection strategy.**

---

## Pseudo-Labeling Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Train U-Net on 258 labeled images                  │
│         → Get supervised baseline (70% F1)                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Use trained U-Net to predict on 4,914 unlabeled    │
│         → Apply class-balanced selection                    │
│         → Generate ~500 high-quality pseudo-labels          │
│         → Particle ratio: 4% (balanced!)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Train NEW U-Net on 258 labeled + 500 pseudo        │
│         → Total: 758 images                                 │
│         → Expected: 73% F1 (+3% improvement!)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Iterate (use improved model for new pseudo-labels) │
│         → Up to 3 iterations                                │
│         → Stop if no improvement                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences from Failed Approach

| Aspect | Naive (0% gain) | Class-Balanced (Expected +3-5%) |
|--------|-----------------|----------------------------------|
| **Model** | ✅ U-Net (31M params) | ✅ U-Net (31M params) |
| **Architecture** | ✅ Same | ✅ Same |
| **Loss function** | ✅ Same (Focal+Dice) | ✅ Same |
| **Optimizer** | ✅ Same (AdamW) | ✅ Same |
| **Pseudo-label selection** | ❌ Naive (single threshold) | ✅ Class-balanced |
| **Particle threshold** | ❌ 0.70 (too high) | ✅ 0.60 (lower) |
| **Background threshold** | ❌ 0.70 (too low) | ✅ 0.80 (higher) |
| **Quantity limit** | ❌ None (3,083 pseudo) | ✅ 2× labeled (516 max) |
| **Quality score** | ❌ Just confidence | ✅ Balance + confidence |

**The model is the same. The pseudo-label selection is different!**

---

## Verification

You can verify the model is correctly instantiated by checking the log:

```bash
grep "MODEL ARCHITECTURE" pseudolabel_6k_balanced_training.log
```

Or count parameters:
```python
from train_unet_selftraining_improved import UNet
model = UNet(in_channels=1, out_channels=1, base_features=64)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Parameters: 31,042,369
```

---

## Summary

✅ **Confirmed**: Using **U-Net** architecture
✅ **Parameters**: 31M (reasonable size)
✅ **Same architecture** across all experiments
✅ **Only difference**: Pseudo-label selection strategy
✅ **Expected improvement**: +3-5% F1 from class-balanced selection

The class-balanced pseudo-labeling experiment is currently running with this exact U-Net architecture!
