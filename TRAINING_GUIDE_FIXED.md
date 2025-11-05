# Fixed PU + FixMatch Training Guide

## Overview

This guide provides a complete solution for training PU + FixMatch on the **full CryoTransformer dataset** (5,172 training images), addressing all issues that caused the previous training to fail.

## What Was Fixed

### 1. **Preprocessing Issues**
- ❌ **Before**: Random CSV-based chunking with poor negative sampling
- ✅ **After**: COCO-based preprocessing using bbox information for smart negative sampling

### 2. **Prior Estimation**
- ❌ **Before**: `prior = 0.3` (wrong! actual was 0.467)
- ✅ **After**: Automatically calculated from data: `prior ≈ 0.47`

### 3. **Learning Rate**
- ❌ **Before**: `lr = 0.0001` (too low for large dataset)
- ✅ **After**: `lr = 0.001` (10x higher)

### 4. **Consistency Threshold**
- ❌ **Before**: `threshold = 0.95` (too high, filtered most pseudo-labels)
- ✅ **After**: `threshold = 0.90` (more balanced)

### 5. **Batch Size**
- ❌ **Before**: `batch_size = 32`
- ✅ **After**: `batch_size = 64` (better gradient estimates)

### 6. **Data Quality**
- ❌ **Before**: Used all images indiscriminately
- ✅ **After**: Smart COCO-based sampling with bbox size awareness

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
./run_full_training.sh
```

This will:
1. Preprocess the full dataset (~10-20 minutes)
2. Calculate the optimal prior from data
3. Train the model with fixed hyperparameters

### Option 2: Manual Step-by-Step

#### Step 1: Preprocessing

```bash
python preprocess_coco_full_dataset.py \
    --data-root /home/uuni/cryoppp/CryoTransformer/train_val_test_data/ \
    --output-dir data/cryotransformer_full_pu \
    --patch-size 128 \
    --hide-ratio 0.3 \
    --neg-per-pos 1.0 \
    --seed 42
```

**Expected output:**
```
Processing train split
Total images: 5172
  Positive patches: ~1,200,000
  Unlabeled patches: ~1,500,000
  Actual prior (π): 0.47XX
```

#### Step 2: Training

```bash
python train_pu_fixed_full.py \
    --config configs/pu_full_fixed.yaml \
    --exp-name pu_full_fixed \
    --device cuda:0 \
    --num-workers 8 \
    --use-data-prior  # Use actual prior from preprocessing
```

## Configuration Details

### `configs/pu_full_fixed.yaml`

```yaml
data:
  batch_ratio: [1, 4]  # 1 positive : 4 unlabeled

model:
  backbone: "resnet18"
  dropout: 0.3

loss:
  prior: 0.47  # Will be overridden by --use-data-prior
  consistency_threshold: 0.90
  lambda_consistency: 1.0

optimizer:
  lr: 0.001  # 10x higher than before
  scheduler: "cosine"

training:
  epochs: 100
  batch_size: 64
  early_stopping_patience: 20
```

## Key Improvements in Code

### 1. **Smart Negative Sampling** (`preprocess_coco_full_dataset.py`)

```python
# Use bbox size to determine minimum distance
avg_particle_size = np.mean(particle_sizes)
min_distance = max(patch_size, avg_particle_size * 1.5)

# Ensure negatives are far from ALL particles
for px, py in particle_centers:
    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
    if dist < min_distance:
        too_close = True
```

### 2. **Proper PU Split**

```python
# Hide 30% of positives in unlabeled set
num_to_hide = int(num_particles * hide_ratio)
keep_indices = indices[num_to_hide:]
hide_indices = indices[:num_to_hide]
```

### 3. **Correct Prior Calculation**

```python
actual_prior = len(positive_patches) / total_patches
# Saved in metadata and used for training
```

### 4. **Enforced Batch Ratio** (`train_pu_fixed_full.py`)

```python
class PUBatchSampler:
    """Enforces exact P:U ratio in each batch"""
    def __init__(self, dataset, batch_size, ratio=[1, 4]):
        self.n_pos_per_batch = int(batch_size * ratio[0] / sum(ratio))
        self.n_unl_per_batch = batch_size - self.n_pos_per_batch
```

## Expected Performance

Based on the small dataset's success (AUC 0.936), we expect:

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Test AUC** | **0.85-0.90** | Full dataset + proper config |
| **Test AP** | **0.90-0.95** | Better than failed attempt (0.58) |
| **Precision** | **0.85-0.92** | Not overpredicting like before |
| **Recall** | **0.75-0.85** | Not 100% like failed model |
| **F1** | **0.80-0.88** | Balanced performance |

### Why These Targets?

1. **Lower than small dataset** (0.936): More data = more noise
2. **Much higher than failed attempt** (0.632): Fixed all major issues
3. **Competitive with CryoTransformer**: Using same data but different approach

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir experiments/
```

Key metrics to watch:
- **Val AUC**: Should increase steadily, plateau around 0.85+
- **Mask Rate**: Should be 60-80% (not too low, not too high)
- **PU Loss**: Should decrease and stabilize
- **Consistency Loss**: Should be non-zero and stable

### Console Output

Good training looks like:
```
Epoch 1/100
  Train Loss: 0.4523
    - PU Loss: 0.2841
    - Consistency Loss: 0.1682
  Consistency Mask Rate: 72.3%

  Val AUC: 0.7234
  Val Precision: 0.7891
  Val Recall: 0.6543

Epoch 20/100
  Train Loss: 0.2134
    - PU Loss: 0.1245
    - Consistency Loss: 0.0889
  Consistency Mask Rate: 78.1%

  Val AUC: 0.8712
  Val Precision: 0.8934
  Val Recall: 0.7823
  *** New best model! AUC: 0.8712 ***
```

### Warning Signs

🚨 **Stop and debug if you see:**
- Mask rate < 20% (threshold too high)
- Mask rate > 95% (threshold too low)
- Val AUC not improving after 10 epochs
- Precision = 100%, Recall = 0% (predicting all negative)
- Precision ~47%, Recall = 100% (predicting all positive)

## Troubleshooting

### Issue: Out of Memory

```bash
# Reduce batch size in config
training:
  batch_size: 32  # or 16
```

### Issue: Training too slow

```bash
# Reduce workers or use smaller subset
python preprocess_coco_full_dataset.py \
    --max-images-train 2000  # Add this parameter (need to modify script)
```

### Issue: Poor performance after training

**Check:**
1. Prior value used (should be ~0.47)
2. Batch ratio enforced (check code)
3. Data preprocessing completed correctly
4. Learning rate not too high/low

## Comparison Table

| Aspect | Failed Attempt | Fixed Version |
|--------|---------------|---------------|
| **Preprocessing** | CSV chunking | COCO-based |
| **Images used** | 5,172 (all) | 5,172 (all) |
| **Prior** | 0.3 ❌ | 0.47 ✅ |
| **Learning rate** | 0.0001 ❌ | 0.001 ✅ |
| **Batch size** | 32 | 64 ✅ |
| **Cons. threshold** | 0.95 ❌ | 0.90 ✅ |
| **Negative sampling** | Random ❌ | Bbox-aware ✅ |
| **Expected AUC** | 0.632 ❌ | 0.85-0.90 ✅ |

## Files Created

```
CryoEM_FixMatch_PU/
├── preprocess_coco_full_dataset.py    # Improved preprocessing
├── train_pu_fixed_full.py             # Fixed training script
├── configs/pu_full_fixed.yaml         # Corrected configuration
├── run_full_training.sh               # Complete pipeline
└── TRAINING_GUIDE_FIXED.md            # This guide
```

## Next Steps After Training

1. **Evaluate on test set**: Results automatically saved in `experiments/*/results.json`

2. **Compare with CryoTransformer**:
   - CryoTransformer uses object detection (DETR)
   - Your approach uses patch classification with PU learning
   - Different strengths for different use cases

3. **Try deeper backbone** (if performance good but want better):
   ```yaml
   model:
     backbone: "resnet34"  # or "resnet50"
   ```

4. **Hyperparameter tuning** (if needed):
   - Adjust `hide_ratio`: Try 0.2 or 0.4
   - Adjust `neg_per_pos`: Try 0.5 or 2.0
   - Adjust `consistency_threshold`: Try 0.85 or 0.92

## Expected Runtime

- **Preprocessing**: 10-20 minutes (one-time)
- **Training**:
  - ~5-10 minutes per epoch (depends on GPU)
  - Early stopping around epoch 20-40
  - Total: 2-6 hours

## Success Criteria

✅ **Training successful if:**
1. Val AUC > 0.80
2. Test AUC > 0.75
3. Precision and Recall both > 0.70
4. Model doesn't collapse (predict all pos or all neg)
5. Performance significantly better than failed attempt (0.632)

---

**Good luck with training! 🚀**

For issues or questions, check:
1. Console output for errors
2. TensorBoard for metrics
3. `experiments/*/results.json` for final numbers
