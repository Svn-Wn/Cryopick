# ⚠️  ROOT CAUSE: Confidence Filtering Imbalance

**Date:** October 15, 2025
**Issue:** FixMatch recall collapse due to imbalanced confidence filtering
**Status:** FIX IN PROGRESS (running with threshold=0.50)

---

## The Problem

Your FixMatch implementation was collapsing recall to nearly zero:

```
Results with confidence_threshold=0.70:
  5%  labels: F1 = 1.68%   (Precision = 67%, Recall = 0.85%)  ❌
  10% labels: F1 = 18.15%  (Precision = 74%, Recall = 10.35%) ❌
```

---

## Root Cause Analysis

### How Confidence Filtering Works:

```python
# From train_ssl_evaluation_fixed.py:419-420
confidence = torch.max(torch.cat([probs_weak, 1 - probs_weak], dim=1), dim=1, keepdim=True)[0]
mask = (confidence >= adaptive_threshold).float()
```

**Translation:**
- **confidence** = max(probability, 1-probability)
- **mask** = keep pixels where confidence ≥ threshold

### The Imbalance:

| Probability | Confidence | Passes threshold 0.70? | Typical for... |
|-------------|------------|------------------------|----------------|
| 0.10 | max(0.10, 0.90) = 0.90 | ✅ YES | Background (confident) |
| 0.20 | max(0.20, 0.80) = 0.80 | ✅ YES | Background (confident) |
| 0.30 | max(0.30, 0.70) = 0.70 | ✅ YES | Background (borderline) |
| 0.40 | max(0.40, 0.60) = 0.60 | ❌ NO  | Uncertain |
| 0.50 | max(0.50, 0.50) = 0.50 | ❌ NO  | Uncertain |
| 0.60 | max(0.60, 0.40) = 0.60 | ❌ NO  | **Particles (filtered!)** |
| 0.70 | max(0.70, 0.30) = 0.70 | ✅ YES | Particles (confident) |
| 0.80 | max(0.80, 0.20) = 0.80 | ✅ YES | Particles (confident) |

### Why This Kills Recall:

In cryo-EM with a **weak model** (56% F1):
- Most **particle predictions** fall in **0.5-0.6 range** → Confidence 0.5-0.6 → **FILTERED OUT**
- Most **background predictions** fall in **0.1-0.3 range** → Confidence 0.7-0.9 → **KEPT**

**Result:** Training data becomes 99% background → Model learns to predict background everywhere → Recall collapses

---

## Diagnostic Results

Running `diagnose_fixmatch_training.py` confirmed the problem:

```
At confidence threshold 0.70:
  → Only keeps 24.2% of particles  ❌
  → Keeps 82.9% of background      ❌
  → Particles in training data: 1.0% (should be ~3-5%)

At confidence threshold 0.50:
  → Keeps 100% of particles  ✅
  → Keeps 100% of background ✅
  → Particles in training data: 3.3% (correct!)
```

### Full Comparison Table:

| Threshold | Masked % | Particles % | Particle Keep % | BG Keep % |
|-----------|----------|-------------|-----------------|-----------|
| 0.50 | 100.0% | 3.3% | 100.0% | 100.0% |
| 0.60 | 95.5% | 1.9% | 56.8% | 96.8% |
| **0.70** | **81.0%** | **1.0%** | **24.2%** | **82.9%** |
| 0.80 | 47.8% | 0.5% | 7.4% | 49.1% |
| 0.90 | 15.1% | 0.3% | 1.5% | 15.6% |

**Key insight:** At threshold 0.70, you filter out **75.8% of particles** but only **17.1% of background**!

---

## The Fix

### Run with Lower Threshold:

```bash
python3 train_ssl_evaluation_fixed.py \
    --image-dir data/cryotransformer/images \
    --coords-file data/cryotransformer/coordinates.json \
    --output-dir experiments/ssl_test_confidence_050 \
    --labeled-ratios 0.05,0.10 \
    --target-size 768 \
    --particle-radius 42 \
    --batch-size 2 \
    --confidence-threshold 0.50  # Changed from 0.70
```

**Status:** ✅ Running now (started background process)

### Expected Results:

| Label Ratio | Before (threshold=0.70) | After (threshold=0.50, Expected) |
|-------------|------------------------|----------------------------------|
| 5% | F1 = 1.68% (Recall = 0.85%) | F1 = ~55-60% (Recall = ~50-55%) |
| 10% | F1 = 18.15% (Recall = 10.35%) | F1 = ~57-62% (Recall = ~53-58%) |

**Expected improvement:** ~35-40× better recall!

---

## Why Confidence Threshold Matters

### FixMatch Original Paper (Classification):

```python
# For multi-class classification
max_prob = torch.max(probs, dim=1)  # Get highest class probability
mask = max_prob >= threshold  # Typically 0.95
```

This works well because:
- All classes are (roughly) balanced
- High confidence truly means the model is certain

### Our Case (Imbalanced Binary Segmentation):

```python
# For binary segmentation
confidence = max(prob, 1-prob)  # PROBLEM: Asymmetric!
mask = confidence >= threshold
```

This FAILS because:
- Classes are **highly imbalanced** (5% particles, 95% background)
- High confidence for background is **much easier** to achieve
- Threshold creates **systematic bias** against rare class (particles)

---

## Alternative Approaches (If 0.50 Doesn't Work)

### Option 1: Even Lower Threshold (0.40 or 0.30)
```bash
--confidence-threshold 0.40
```

### Option 2: Disable Confidence Filtering (threshold=0.0)
```bash
--confidence-threshold 0.0
```
This uses **all pseudo-labels**, which may be noisy but avoids imbalance.

### Option 3: Different Confidence Metric
Modify line 419 in `train_ssl_evaluation_fixed.py`:
```python
# CURRENT (asymmetric):
confidence = torch.max(torch.cat([probs_weak, 1 - probs_weak], dim=1), dim=1, keepdim=True)[0]

# OPTION A: Use probability directly (symmetric for both classes):
confidence = torch.abs(probs_weak - 0.5) + 0.5  # Range: 0.5-1.0
# prob=0.1 → conf=0.9, prob=0.5 → conf=0.5, prob=0.9 → conf=0.9

# OPTION B: Use probability with class-aware threshold:
# Keep particle pixels with prob >= 0.60
# Keep background pixels with prob <= 0.40
mask_particles = (probs_weak >= 0.60).float()
mask_background = (probs_weak <= 0.40).float()
mask = mask_particles + mask_background
```

### Option 4: Confidence Weighting Without Masking
Remove the hard threshold, just use soft weights:
```python
# Remove mask, use confidence as weight directly
weights = confidence  # Don't threshold
loss_consistency = (loss_consistency * weights).mean()
```

---

## Timeline

### Current Run (threshold=0.50):
- **Started:** October 15, 2025
- **Duration:** ~2-3 hours
- **Experiments:** 5% and 10% labels
- **Output:** `experiments/ssl_test_confidence_050/`

### Check Progress:
```bash
# Watch output directory
watch -n 10 'ls -lh experiments/ssl_test_confidence_050/'

# Check GPU
nvidia-smi

# View results when ready
cat experiments/ssl_test_confidence_050/ssl_eval_fixmatch_ratio_5.json
cat experiments/ssl_test_confidence_050/ssl_eval_fixmatch_ratio_10.json
```

---

## Summary

### What Went Wrong:
1. Confidence threshold 0.70 was too high for a weak model
2. Confidence calculation favors background over particles
3. At threshold 0.70: Only 24% of particles kept, 83% of background kept
4. Extreme imbalance in training data → Model learns background only → Recall collapses

### What We Fixed:
1. Lowered confidence threshold from 0.70 → 0.50
2. Now keeps 100% of both particles and background
3. Training data should have proper class balance
4. Recall should improve from 1-10% to 50-60%

### What to Expect:
- ✅ Better class balance in pseudo-labels
- ✅ Recall should improve dramatically (~40× better)
- ✅ F1 should be positive gain instead of -54% degradation
- ⚠️  If still poor, try threshold 0.40 or disable filtering (0.0)

---

**Wait for training to complete, then check results!** 🚀
