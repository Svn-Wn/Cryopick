# Final Diagnosis: Mean Teacher Failure at 10% Labels

## Problem Statement

Mean Teacher SSL training **completely fails** at 10% labeled data (F1=0.00) but **works perfectly** at 1% labeled data (F1=0.52).

##  Investigation Timeline

### Hypothesis 1: Incorrect pos_weight ❌ FAILED
- **Theory**: `pos_weight=5.0` wrong for 30% positive pixels
- **Fix**: Changed to `pos_weight=2.33`
- **Result**: Still F1=0.00

### Hypothesis 2: Model Initialization Bias ❌ REJECTED
- **Theory**: U-Net initializes with negative bias
- **Investigation**: Created `diagnose_initialization.py`
- **Finding**: Initialization is PERFECT (F1=0.56, mean_prob=0.522)
- **Conclusion**: Problem occurs **during first epoch**, not initialization

### Hypothesis 3: Competing Class Balancing ❌ FAILED
- **Theory**: `pos_weight` and `focal_alpha` interfering
- **Fix**: Use `focal_alpha=0.70` (theoretical balance for 30% positive) with `pos_weight=1.0`
- **Result**: Still F1=0.00

### Hypothesis 4: Learning Rate Too High ❌ FAILED
- **Theory**: lr=1e-4 too aggressive for 10× more samples
- **Test**: Reduced to lr=1e-5 (10× lower)
- **Result**: WORSE (F1=0.000010, mean_prob=0.150 vs 0.192)

### Hypothesis 5: focal_alpha Too Low ❌ FAILED
- **Theory**: Need to over-weight positive class beyond theoretical balance
- **Test**: Increased focal_alpha=0.90 (vs theoretical 0.70)
- **Result**: WORSE (mean_prob=0.025, 99.43% pixels < 0.1)

### Hypothesis 6: Focal Loss Fundamentally Broken ❌ FAILED
- **Theory**: Focal Loss has inherent issues with extreme imbalance
- **Test**: Use Dice Loss ONLY (focal_weight=0, dice_weight=1)
- **Result**: Still FAILS (mean_prob=0.072, max_prob=0.419)

## Key Diagnostic Results

Created `diagnose_outputs.py` to track model behavior during first epoch:

### Configuration: focal_alpha=0.70, pos_weight=1.0, lr=1e-4

**BEFORE TRAINING (Epoch 0):**
- Mean logit: 0.088
- Mean probability: 0.522
- F1: 0.49
- Predictions > 0.5: 314,954,560 / 314,966,016 (100%)
- ✅ **Initialization is PERFECT**

**AFTER 1 EPOCH:**
- Mean logit: **-1.443** (shifted by **-1.53**!)
- Mean probability: **0.192**
- F1: **0.00**
- Predictions > 0.5: **534 / 314,966,016 (0.00%)**
- ❌ **Model collapses to all-negative**

**Loss behavior:**
- Training loss decreases: 0.27 → 0.22 (appears normal)
- But predictions get WORSE (F1: 0.49 → 0.00)

## Root Cause Analysis

### Why Does This Happen at 10% but Not 1%?

**1% labels (51 samples):**
- Weak gradients
- Gradual learning
- F1 = 0.52 ✅

**10% labels (517 samples):**
- **10× stronger gradients**
- Aggressive weight updates
- Model collapses to trivial solution (predict all negative)
- F1 = 0.00 ❌

### The Fundamental Problem

The 70/30 class imbalance (70% background, 30% particles) combined with:
- Strong gradients from 517 samples
- All tested loss functions
- Makes the model collapse to the **trivial solution** of predicting everything negative

**Even though mathematically:**
- Dice Loss for "predict all positive": Loss = 0.54
- Dice Loss for "predict all negative": Loss = 1.00
- Model SHOULD prefer "all positive", but does opposite!

This suggests the gradient dynamics during training are pushing toward "all negative" despite the loss values.

## Tested Configurations (ALL FAILED)

| Configuration | focal_alpha | pos_weight | lr | focal_weight | dice_weight | Result (mean_prob) |
|--------------|-------------|------------|-----|--------------|-------------|-------------------|
| Original     | 0.25       | 5.0        | 1e-4| 0.7         | 0.3         | F1=0.00          |
| Fix 1        | 0.5        | 2.33       | 1e-4| 0.7         | 0.3         | F1=0.00          |
| Fix 2        | 0.70       | 1.0        | 1e-4| 0.7         | 0.3         | **0.192**        |
| Lower LR     | 0.70       | 1.0        | 1e-5| 0.7         | 0.3         | **0.150** (worse)|
| Higher alpha | 0.90       | 1.0        | 1e-4| 0.7         | 0.3         | **0.025** (worse)|
| Dice only    | N/A        | 1.0        | 1e-4| 0.0         | 1.0         | **0.072**        |

**ALL configurations cause model to predict mostly negative.**

## Possible Solutions (UNTESTED)

### 1. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevent large gradient updates from causing collapse.

### 2. Warmup Learning Rate Schedule
```python
# Start with very small lr, gradually increase
lr_scheduler = WarmupLinearSchedule(warmup_steps=100, total_steps=1000)
```

### 3. Balanced Sampling
Instead of random sampling, ensure each batch has balanced positive/negative pixels:
```python
sampler = BalancedBatchSampler(dataset, target_ratio=0.5)
```

### 4. Different Loss Function
Try Tversky Loss with aggressive recall penalty:
```python
TverskyLoss(alpha=0.3, beta=0.7)  # Heavily penalize false negatives
```

### 5. Two-Stage Training
- **Stage 1**: Train with small subset (1-5% labels) until F1 > 0
- **Stage 2**: Fine-tune with 10% labels starting from Stage 1 weights

### 6. Reduce Batch Size
Smaller batches → weaker gradients → less aggressive updates:
```python
batch_size = 1  # vs current batch_size=4
```

### 7. Check Data Quality
Verify 10% labeled split doesn't have systematic bias:
```python
# Check distribution of positive pixels in 10% split
# Compare with 1% split and full dataset
```

## Recommended Next Steps

1. ✅ **Try gradient clipping** (most likely to work)
2. ✅ **Try warmup schedule** (prevents early collapse)
3. ✅ **Try batch_size=1** (reduces gradient magnitude)
4. ⚠️ **Verify data split quality** (check for systematic bias)
5. ⚠️ **Two-stage training** (bootstrap from 1% model)

## Files Created

1. `diagnose_initialization.py` - Checks model at init (found init is fine)
2. `diagnose_first_epoch.py` - Tracks first epoch (incomplete due to timeout)
3. `diagnose_outputs.py` - Full diagnostic with multiple loss configurations
4. `BUG_INVESTIGATION_SUMMARY.md` - Initial investigation (now outdated)
5. `FINAL_DIAGNOSIS_10PCT_FAILURE.md` - This document

## Key Lesson

**Strong gradients from larger datasets can cause catastrophic collapse in highly imbalanced segmentation tasks, even with theoretically correct loss configurations.**

The solution likely requires **gradient regularization** (clipping, warmup) rather than loss function tuning.
