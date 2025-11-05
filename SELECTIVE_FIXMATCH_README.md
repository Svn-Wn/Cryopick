# Selective FixMatch Implementation for CryoEM Particle Picking

## Overview

This is a complete implementation of **Selective FixMatch** for semi-supervised learning on CryoEM particle picking. Unlike traditional PU learning approaches, this method uses:

1. **Supervised BCE loss** on positive samples (with weak augmentation)
2. **Consistency loss** on high-confidence unlabeled samples (weak + strong augmentation)
3. **EMA teacher model** for stable pseudo-labeling

No PU loss, no false-positive branches - just clean semi-supervised learning.

---

## Architecture

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Selective FixMatch                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Positive Samples → Weak Aug → Model → L_sup (BCE)          │
│                                                              │
│  Unlabeled Samples:                                          │
│    → Weak Aug → Model → Pseudo-labels (if conf >= τ)        │
│    → Strong Aug → Model → L_cons (consistency)               │
│                                                              │
│  Total Loss: L_total = L_sup + λ * L_cons                   │
│                                                              │
│  EMA Model: θ_ema ← α * θ_ema + (1-α) * θ_model             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

- **Backbone**: ResNet18 (11.2M parameters)
- **Input**: 1-channel grayscale images (128×128)
- **Output**: Binary logits (before sigmoid)
- **EMA decay**: 0.999

### Augmentation Strategy

**Weak Augmentation** (both positive and unlabeled):
- Random horizontal/vertical flip
- Random rotation (±10°)
- Minor Gaussian noise (σ=0.01-0.05)

**Strong Augmentation** (unlabeled only):
- Random horizontal/vertical flip
- Large rotations (90°, 180°, 270°)
- Strong contrast/brightness jitter (±30%)
- Strong Gaussian noise (σ=0.1-0.2) or Poisson noise
- Light Gaussian blur
- Random erasing (small patches)

### Loss Function

```
L_total = L_sup + λ(epoch) * L_cons

where:
- L_sup: BCE loss on positive samples
- L_cons: Consistency loss on high-confidence unlabeled samples
- λ(epoch): Linearly ramped from 0 to 1.0 over first 10 epochs
```

**Confidence Threshold**: τ = 0.8 (only use pseudo-labels with confidence >= 80%)

---

## File Structure

```
datasets/
├── cryoem_dataset_selectivefixmatch.py   # Dataset loader
└── transforms_cryoem_selectivefixmatch.py # Augmentation transforms

models/
├── selective_fixmatch_resnet.py          # ResNet18 + EMA model
└── losses_selectivefixmatch.py           # Selective FixMatch loss

configs/
└── selective_fixmatch.yaml                # Hyperparameters

train_selective_fixmatch.py               # Training script
```

---

## Usage

### 1. Quick Test (Recommended First Step)

Test individual components:

```bash
# Test loss function
python models/losses_selectivefixmatch.py

# Test model
python models/selective_fixmatch_resnet.py

# Test transforms
cd datasets && python transforms_cryoem_selectivefixmatch.py && cd ..
```

### 2. Training

**Default training** (ResNet18, 50 epochs):

```bash
python train_selective_fixmatch.py \
    --config configs/selective_fixmatch.yaml \
    --exp-name selective_fixmatch \
    --device cuda:0
```

**Background training** (recommended):

```bash
nohup python train_selective_fixmatch.py \
    --config configs/selective_fixmatch.yaml \
    --exp-name selective_fixmatch \
    --device cuda:0 \
    > train_selective_fixmatch.log 2>&1 &
```

**Monitor progress**:

```bash
# Watch training log
tail -f train_selective_fixmatch.log

# Check GPU usage
nvidia-smi -l 1
```

### 3. Configuration Options

Edit `configs/selective_fixmatch.yaml` to customize:

```yaml
dataset:
  batch_size: 64           # Batch size
  num_workers: 8           # Data loading workers
  load_into_ram: true      # Load all data into RAM (faster)

model:
  backbone: "resnet18"     # resnet18, resnet34, resnet50
  dropout: 0.3             # Dropout rate
  ema_decay: 0.999         # EMA decay rate

loss:
  consistency_weight: 1.0      # λ (consistency weight)
  confidence_threshold: 0.8    # τ (confidence threshold)
  rampup_epochs: 10            # Ramp-up period for λ

optimizer:
  lr: 0.001                # Learning rate
  weight_decay: 0.0001     # Weight decay

training:
  epochs: 50               # Total epochs
  early_stopping_patience: 15  # Early stopping patience
```

---

## Expected Results

### Training Metrics

Typical training progress:

```
Epoch 1 | L_sup=0.65 | L_cons=0.00 | Mask=0% | AUC=0.78
Epoch 5 | L_sup=0.42 | L_cons=0.18 | Mask=25% | AUC=0.85
Epoch 10 | L_sup=0.31 | L_cons=0.12 | Mask=35% | AUC=0.89
Epoch 20 | L_sup=0.25 | L_cons=0.08 | Mask=40% | AUC=0.92
Epoch 30 | L_sup=0.21 | L_cons=0.06 | Mask=42% | AUC=0.93
```

### Final Performance Goals

- **AUC**: > 0.90 (excellent discrimination)
- **F1**: > 0.85 (good balance of precision and recall)
- **Mask Rate**: 30-40% (good pseudo-label utilization)

---

## Key Differences from PU Learning

| Aspect | PU Learning (nnPU, GE-binomial) | Selective FixMatch |
|--------|--------------------------------|-------------------|
| **Loss on positives** | Weighted BCE | Standard BCE |
| **Loss on unlabeled** | PU risk estimator | Consistency loss |
| **Prior needed?** | Yes (π=0.413) | No |
| **Complexity** | High (complex math) | Low (simple BCE) |
| **Stability** | Can be unstable | Very stable |
| **Proven on** | Text classification | Image classification |

---

## Troubleshooting

### Low mask rate (< 10%)

- Model predictions too uncertain
- **Solution**: Lower `confidence_threshold` to 0.7 or 0.6

### High mask rate (> 60%)

- Model overconfident (possible overfitting)
- **Solution**: Increase dropout or reduce epochs

### AUC not improving

- Check if L_sup is decreasing (model learning on positives)
- Try lower learning rate (0.0005)
- Check data loading (ensure positives are labeled correctly)

### Out of memory

- Reduce `batch_size` to 32 or 16
- Set `load_into_ram: false` in config

---

## Output Files

After training, results are saved to `experiments/{exp_name}/`:

```
experiments/selective_fixmatch/
├── best_model.pt           # Best student model (by val AUC)
├── best_model_ema.pt       # Best EMA teacher model
├── last_checkpoint.pt      # Latest checkpoint
└── results.json            # Full training history + test metrics
```

---

## Citation

This implementation combines ideas from:

1. **FixMatch** (Sohn et al., NeurIPS 2020)
2. **Mean Teacher** (Tarvainen & Valpola, NeurIPS 2017)
3. **CryoEM Particle Picking** (Topaz, Nature Methods 2019)

---

## Contact

For issues or questions about this implementation, refer to:
- FixMatch paper: https://arxiv.org/abs/2001.07685
- Mean Teacher: https://arxiv.org/abs/1703.01780

