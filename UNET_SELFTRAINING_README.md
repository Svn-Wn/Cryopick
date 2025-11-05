# U-Net Self-Training for Cryo-EM Particle Picking

Complete implementation of iterative self-training (Selective FixMatch) using U-Net for semantic segmentation of particles in cryo-EM micrographs.

---

## Overview

This implementation uses **U-Net** for semantic segmentation combined with **self-training** to handle positive-unlabeled (PU) learning scenarios.

### Key Features:

1. ✅ **U-Net Architecture**: Standard encoder-decoder with skip connections
2. ✅ **Iterative Self-Training**: Progressively refines predictions using pseudo-labels
3. ✅ **Selective Pseudo-Labeling**: Uses confidence thresholds for high-quality labels
4. ✅ **Ignore Mask Support**: Handles uncertain regions during training
5. ✅ **Configurable Pipeline**: All hyperparameters easily adjustable

---

## How It Works

### Phase 1: Initial Supervised Training

1. Convert particle coordinates → binary segmentation masks (circles)
2. Train U-Net on labeled particles vs background
3. Save baseline model

### Phase 2: Iterative Self-Training Loop

**For each iteration:**

1. **Inference**: Generate prediction probability maps for all images
2. **Pseudo-Labeling**:
   - High confidence (>0.95) → Pseudo-positive
   - Low confidence (<0.05) → Reliable-negative
   - Medium confidence → Ignored
3. **Label Combination**:
   - Original labels (highest priority)
   - Pseudo-positive labels
   - Reliable-negative labels
   - Uncertain pixels ignored
4. **Retraining**: Fine-tune model on combined labels

This iterative process progressively improves the model by leveraging unlabeled data.

---

## Installation

### Requirements:

```bash
pip install torch torchvision opencv-python numpy tqdm
```

### Files:

- `train_unet_selftraining.py` - Main training script
- `unet_data_adapter.py` - Converts chunked dataset to U-Net format
- `run_unet_selftraining.sh` - Quick start script

---

## Usage

### Option 1: Using Existing Chunked Dataset

**Step 1: Convert data to U-Net format**

```bash
python unet_data_adapter.py \
  --chunk-dir /mnt/hdd1/uuni/fixmatch/cryotransformer_subset_pu \
  --output-dir data/unet_format \
  --split train \
  --particle-radius 10 \
  --max-samples 1000  # Optional: limit for testing
```

**Step 2: Run self-training**

```bash
python train_unet_selftraining.py \
  --image-dir data/unet_format/images \
  --coords-file data/unet_format/coordinates.json \
  --output-dir experiments/unet_selftraining \
  --particle-radius 10 \
  --initial-epochs 50 \
  --self-training-iterations 5 \
  --retrain-epochs 20 \
  --learning-rate 0.001 \
  --batch-size 8 \
  --positive-threshold 0.95 \
  --negative-threshold 0.05 \
  --device cuda:0
```

### Option 2: Using Custom Data

**Required format:**

1. **Images**: Directory with `.png` or `.tif` files
2. **Coordinates JSON**:
```json
{
  "image1.png": [[x1, y1], [x2, y2], ...],
  "image2.png": [[x3, y3], ...],
  ...
}
```

**Run training:**

```bash
python train_unet_selftraining.py \
  --image-dir /path/to/images \
  --coords-file /path/to/coordinates.json \
  --output-dir experiments/unet_custom \
  --particle-radius 15 \
  --initial-epochs 100 \
  --self-training-iterations 3 \
  --retrain-epochs 30
```

---

## Configuration Parameters

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--particle-radius` | Radius of particles (pixels) for mask generation | 10 | 10-20 |
| `--initial-epochs` | Epochs for initial supervised training | 50 | 50-100 |
| `--self-training-iterations` | Number of self-training loops | 5 | 3-10 |
| `--retrain-epochs` | Epochs per self-training iteration | 20 | 10-30 |
| `--learning-rate` | Optimizer learning rate | 0.001 | 0.0001-0.01 |
| `--batch-size` | Training batch size | 4 | 4-16 |

### Pseudo-Labeling Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--positive-threshold` | Confidence for pseudo-positive labels | 0.95 | 0.90-0.98 |
| `--negative-threshold` | Confidence for reliable-negative labels | 0.05 | 0.02-0.10 |

**Key Trade-offs:**
- **Higher thresholds** → Fewer but higher quality pseudo-labels
- **Lower thresholds** → More pseudo-labels but potentially noisier

---

## Output Files

After training, the following files are saved:

```
experiments/unet_selftraining/
├── baseline_model.pt              # Model after initial supervised training
├── model_iteration_1.pt           # Model after 1st self-training iteration
├── model_iteration_2.pt           # Model after 2nd self-training iteration
├── ...
├── final_model.pt                 # Final trained model
└── training_summary.json          # Training configuration and statistics
```

---

## Architecture Details

### U-Net Structure

```
Input (1, H, W)
    ↓
Encoder:
  - Conv → BN → ReLU → Conv → BN → ReLU → MaxPool (×4)
  - Features: 64 → 128 → 256 → 512
    ↓
Bottleneck:
  - Conv → BN → ReLU → Conv → BN → ReLU
  - Features: 1024
    ↓
Decoder:
  - Upsample → Concat(skip) → Conv → BN → ReLU (×4)
  - Features: 512 → 256 → 128 → 64
    ↓
Output (1, H, W) - Probability map
```

**Total parameters**: ~31M (base_features=64)

---

## Loss Function

### IgnoreMaskBCELoss

Binary Cross-Entropy with support for ignore masks:

```python
mask_values:
  -1 → Ignored (uncertain regions)
   0 → Background
   1 → Particle

loss = BCE(predictions, targets) * valid_mask
```

Only computes loss over non-ignored pixels.

---

## Self-Training Algorithm

### Pseudo-Code

```python
# Phase 1: Initial Training
masks = coordinates_to_masks(coords, radius)
model = train(images, masks, epochs=50)

# Phase 2: Self-Training Loop
for iteration in range(5):
    # Inference
    predictions = model.predict(images)

    # Pseudo-labeling
    pseudo_pos = predictions > 0.95
    reliable_neg = predictions < 0.05

    # Combine labels
    combined_masks = combine(
        original=masks,
        pseudo_positive=pseudo_pos,
        reliable_negative=reliable_neg
    )

    # Retrain
    model = train(images, combined_masks, epochs=20)
```

---

## Monitoring Training

### During Training

Watch for these metrics in console output:

```
Phase 1: Initial Supervised Training
Epoch 1/50 - Loss: 0.3456
Epoch 2/50 - Loss: 0.2891
...

Self-Training Iteration 1/5
A. Running inference...
B. Generating pseudo-labels...
   Avg pseudo-positive coverage: 15.32%
   Avg reliable-negative coverage: 72.45%
C. Combining labels...
   Positive pixels: 18.67%
   Negative pixels: 75.21%
   Ignored pixels: 6.12%
D. Retraining...
   Epoch 1/20 - Loss: 0.2103
   ...
```

### Key Indicators

✅ **Healthy Training**:
- Loss decreasing over iterations
- Pseudo-positive coverage: 10-30%
- Reliable-negative coverage: 60-80%
- Ignored pixels: <20%

⚠️ **Warning Signs**:
- Loss increasing → Lower learning rate
- Pseudo-positive >50% → Increase positive threshold
- Ignored pixels >50% → Widen threshold gap

---

## Tips & Best Practices

### 1. Particle Radius Selection

```bash
# Too small → Misses particle boundaries
--particle-radius 5

# Good balance
--particle-radius 10-15

# Too large → Includes background
--particle-radius 30
```

### 2. Threshold Tuning

Start conservative, then relax:

```bash
# Conservative (fewer but cleaner pseudo-labels)
--positive-threshold 0.98 --negative-threshold 0.02

# Balanced (recommended)
--positive-threshold 0.95 --negative-threshold 0.05

# Aggressive (more pseudo-labels)
--positive-threshold 0.90 --negative-threshold 0.10
```

### 3. Iteration Strategy

**Quick Test** (3-5 iterations):
```bash
--initial-epochs 30 \
--self-training-iterations 3 \
--retrain-epochs 10
```

**Full Training** (5-10 iterations):
```bash
--initial-epochs 100 \
--self-training-iterations 7 \
--retrain-epochs 30
```

---

## Comparison with ResNet FixMatch

| Aspect | U-Net Self-Training | ResNet FixMatch |
|--------|---------------------|-----------------|
| **Task** | Semantic segmentation | Binary classification |
| **Output** | Pixel-wise probability map | Single prediction per patch |
| **Pseudo-labels** | Confidence-based regions | High-confidence samples |
| **Spatial info** | Preserves full spatial context | Patch-level only |
| **Model size** | ~31M params | ~11M params |
| **Training time** | Slower (dense predictions) | Faster (single output) |

**When to use U-Net:**
- Need precise particle localization
- Want probability maps for visualization
- Have full micrographs (not just patches)

**When to use ResNet:**
- Patch-level classification sufficient
- Need faster training/inference
- Limited GPU memory

---

## Troubleshooting

### Issue: Loss not decreasing

**Solution:**
```bash
# Reduce learning rate
--learning-rate 0.0001

# Or increase initial training
--initial-epochs 100
```

### Issue: Too many pseudo-positives (model collapse)

**Solution:**
```bash
# Increase positive threshold
--positive-threshold 0.98

# Add more initial supervised training
--initial-epochs 150
```

### Issue: GPU Out of Memory

**Solution:**
```bash
# Reduce batch size
--batch-size 2

# Or reduce U-Net features
# Edit train_unet_selftraining.py:
# UNet(base_features=32)  # Instead of 64
```

---

## Example: Quick Test Run

**Test on small subset (10 minutes):**

```bash
# 1. Convert data (1000 samples)
python unet_data_adapter.py \
  --chunk-dir /mnt/hdd1/uuni/fixmatch/cryotransformer_subset_pu \
  --output-dir data/unet_test \
  --max-samples 1000

# 2. Run self-training (minimal config)
python train_unet_selftraining.py \
  --image-dir data/unet_test/images \
  --coords-file data/unet_test/coordinates.json \
  --output-dir experiments/unet_test \
  --initial-epochs 10 \
  --self-training-iterations 2 \
  --retrain-epochs 5 \
  --batch-size 8 \
  --device cuda:0
```

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{sohn2020fixmatch,
  title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
  author={Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin A and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang},
  journal={Advances in neural information processing systems},
  year={2020}
}
```

---

## Advanced: Inference Script

To use trained model for inference:

```python
import torch
from train_unet_selftraining import UNet
import numpy as np

# Load model
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('experiments/unet_selftraining/final_model.pt'))
model.eval()

# Inference
image = np.load('test_image.npy')  # (H, W)
image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

with torch.no_grad():
    logits = model(image_tensor)
    probs = torch.sigmoid(logits).squeeze().numpy()

# Threshold to get binary mask
binary_mask = (probs > 0.5).astype(np.uint8)
```

---

## Questions?

For issues or questions:
1. Check training logs for error messages
2. Review configuration parameters
3. Try quick test run first
4. Check GPU memory usage

---

*Last updated: Self-training U-Net implementation for CryoEM particle picking*
