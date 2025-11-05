# Debugging Report: CryoEM FixMatch + PU Learning

## Issues Identified & Solutions

### 1. ✅ Normalization Consistency
**Finding:** Normalization is relatively consistent across splits (mean diff: 0.0087, std diff: 0.0056)
- Train mean: 0.4939, std: 0.1724
- Val mean: 0.4852, std: 0.1780
- Test mean: 0.4905, std: 0.1739

**Status:** NOT the main issue, but can be improved

**Fix Provided:**
- Created `preprocessing_fixed.py` with `ConsistentNormalizer` class
- Fits normalization parameters on training data only
- Applies same parameters to all splits

### 2. ⚠️ Pseudo-Label Threshold (Critical Issue)
**Finding:** With τ=0.95, only 1% of unlabeled samples contribute!
- τ=0.95: 1% used
- τ=0.80: 23% used
- τ=0.75: 34% used

**Status:** MAJOR ISSUE - prevents semi-supervised learning

**Fix Applied in `configs/fixmatch_pu_debugged.yaml`:**
```yaml
consistency_threshold: 0.75  # Was 0.95
```

### 3. ⚠️ PU Loss Prior (Critical Issue)
**Finding:** Prior mismatch
- Config uses: π = 0.3
- Actual data: π = 0.197 (490 positive / 2490 P+U)

**Status:** MAJOR ISSUE - causes model bias

**Fix Applied in `configs/fixmatch_pu_debugged.yaml`:**
```yaml
prior: 0.197  # Was 0.3
```

### 4. 🔴 Additional Issue Found: Learning Rate Too High
**Finding:** Learning rate 0.001 is too high for noisy data
**Fix Applied:**
```yaml
lr: 0.00005  # Was 0.001
```

### 5. 🔴 Additional Issue: Loss Weights Imbalanced
**Finding:** Consistency loss weight too high for early training
**Fix Applied:**
```yaml
lambda_consistency: 0.3  # Was 1.0
rampup_epochs: 30  # Was 10
```

## Why Val AUC Still Around 0.5

Even with fixes, the model may need:

1. **More training data**: Only 343 positive samples in training
2. **Better initialization**: Pre-train on synthetic data
3. **Architecture changes**: ResNet18 may be too large
4. **Loss function adjustment**: The PU loss implementation may need review

## Recommended Next Steps

1. **Verify pseudo-labels are being generated:**
```python
# Add to training loop
print(f"Pseudo-label mask rate: {metrics.get('mask_rate', 0):.2%}")
```

2. **Monitor gradient flow:**
```python
# Check if model is actually updating
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

3. **Try simpler baseline:**
- Train with supervised loss only (no FixMatch) on labeled data
- If this also fails, the issue is more fundamental

4. **Data quality check:**
- Visualize positive vs negative patches
- Ensure labels are correct
- Check for data leakage between splits

## Code to Apply All Fixes

```bash
# Use the debugged configuration
python train_fixed.py \
  --config configs/fixmatch_pu_debugged.yaml \
  --exp-name fully_debugged \
  --num-workers 0
```

## Expected Improvements

With these fixes, you should see:
- ✅ More unlabeled samples contributing (check mask_rate > 0.2)
- ✅ Gradual AUC improvement (should reach 0.6+ within 20 epochs)
- ✅ Lower but more stable loss values
- ✅ Better convergence

If AUC remains at 0.5 after these fixes, the issue is likely in:
1. Data quality/labeling
2. Model architecture (try smaller model)
3. PU loss implementation (may need debugging)