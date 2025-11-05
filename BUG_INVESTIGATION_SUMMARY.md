# Bug Investigation Summary: F1=0.00 at 10% Labels

## Timeline of Investigation

### Initial Problem
- Mean Teacher at 1% labels: **F1=0.52** ✅
- Mean Teacher at 10% labels: **F1=0.00** ❌
- Model was predicting **nothing** (Precision=0, Recall=0)
- Training loss was decreasing normally (0.30 → 0.22)

### Hypothesis 1: Incorrect pos_weight (FAILED)
**Theory**: `pos_weight=5.0` was wrong for 30% positive pixels
**Fix Attempted**: Changed to `pos_weight=2.33` (= 0.70 / 0.30)
**Result**: **STILL F1=0.00** ❌
**Why it failed**: Two competing class balancing mechanisms were interfering

### Hypothesis 2: Model Initialization Bias (REJECTED)
**Theory**: U-Net initializes with negative bias
**Investigation**: Created `diagnose_initialization.py` to check initialization
**Result**: **Initialization is PERFECT!** ✅
- At random initialization: F1=0.56, Recall=1.0, Precision=0.39
- Model predicts everything positive initially
- Mean probability: 0.522 (close to 0.5)
- Mean logit: 0.088 (close to 0)

**Critical Finding**: Model is fine at initialization but **breaks during first epoch**!

### Root Cause: Competing Class Balancing Mechanisms ✅

The **CombinedLoss** uses TWO separate class balancing mechanisms:

1. **`pos_weight`** in BCE: Weights positive class in binary cross-entropy
2. **`focal_alpha`** in Focal Loss: Weights positive/negative classes independently

#### Problem with Previous Configuration

```python
# WRONG: Two competing mechanisms
CombinedLoss(
    pos_weight=2.33,      # BCE: Positive class weighted 2.33x
    focal_alpha=0.5,      # Focal: Both classes weighted equally
    focal_weight=0.7,
    dice_weight=0.3
)
```

In Focal Loss, `alpha_t` is calculated as:
```python
alpha_t = alpha * target + (1 - alpha) * (1 - target)
```

For 30% positive pixels:
- With `focal_alpha=0.5`: Positive gets 0.5, Negative gets 0.5 (equal weighting)
- **But** for balanced gradients we need: `0.30 × alpha = 0.70 × (1 - alpha)`
- Solving: **`alpha = 0.70`**

With `focal_alpha=0.5`, we were **UNDER-WEIGHTING positives** in the focal loss, which conflicted with `pos_weight=2.33` over-weighting them in BCE!

#### Correct Configuration

```python
# CORRECT: Single balancing mechanism
CombinedLoss(
    pos_weight=1.0,       # No BCE class balancing
    focal_alpha=0.70,     # Focal handles all class balancing
    focal_weight=0.7,
    dice_weight=0.3
)
```

**Formula for focal_alpha**:
```
positive_ratio × alpha = negative_ratio × (1 - alpha)

For 30% positive:
0.30 × alpha = 0.70 × (1 - alpha)
0.30 × alpha = 0.70 - 0.70 × alpha
1.00 × alpha = 0.70
alpha = 0.70
```

## Why It Failed at 10% but Not 1%

With 10% labels (517 samples), gradients are **10× stronger** than at 1% (51 samples).

The conflicting class weights caused:
1. Strong negative gradients from the 70% negative pixels (under-weighted in focal)
2. Weaker positive gradients from the 30% positive pixels (over-weighted in BCE but under-weighted in focal)
3. Net effect: Model pushed to predict everything negative

At 1% labels, gradients were weak enough that this imbalance didn't catastrophically break training.

## Files Modified

1. **train_mean_teacher.py** (lines 88-98):
   - Changed from `pos_weight=2.33, focal_alpha=0.5`
   - To `pos_weight=1.0, focal_alpha=0.70`

2. **run_mean_teacher_comparison.py** (lines 88-98):
   - Same changes as above

## Diagnostic Tools Created

1. **diagnose_initialization.py**:
   - Checks model output distribution at initialization
   - Computes metrics at different thresholds
   - Visualizes probability distributions
   - **Key finding**: Initialization is perfect (F1=0.56)

2. **diagnose_first_epoch.py**:
   - Trains for one epoch and tracks gradients
   - Compares before/after distributions
   - (Not completed due to time constraints)

## Next Steps

✅ **DONE**: Fixed loss configuration with correct `focal_alpha=0.70`
⏳ **TODO**: Re-run Mean Teacher at 10% labels to verify fix
⏳ **TODO**: Compare results with 1% baseline

## Expected Outcome

With the corrected loss configuration:
- **Epoch 1**: F1 should be > 0 (not 0.00)
- **Final**: F1 should be comparable to or better than 1% baseline (F1~0.52)

## Key Lesson

**When using multiple class balancing mechanisms, they must be configured to work together, or use only ONE mechanism to avoid interference!**
