# ✅ Fixed: Confidence Threshold Too High

**Date:** October 15, 2025
**Issue:** FixMatch causing performance degradation due to recall collapse
**Status:** FIXED

---

## Problem Diagnosed

### Symptoms:
```
10% labels: Supervised F1 = 56.86%, FixMatch F1 = 18.14% (-38.72%)

FixMatch Metrics:
  Precision = 70.95% (good)
  Recall    = 10.40% (TERRIBLE)
  → RECALL COLLAPSE
```

### Root Cause:
**Confidence threshold was hardcoded at 0.95**, which is too high for cryo-EM particle detection.

```python
# OLD (train_ssl_evaluation_fixed.py:280)
adaptive_threshold = max(0.95, supervised_precision)
                    = max(0.95, 0.5689)
                    = 0.95  # TOO HIGH!
```

**Effect:**
- At 95% confidence: Very few pixels get pseudo-labels
- Model becomes overly conservative
- High precision (70%) but terrible recall (10%)
- Overall F1 collapses to 18%

---

## Solution Implemented

### Changes Made:

1. **Added `--confidence-threshold` parameter** (line 743):
```python
parser.add_argument('--confidence-threshold', type=float, default=0.70,
                   help='Confidence threshold for pseudo-labels (default: 0.70)')
```

2. **Updated function signature** (line 521):
```python
def run_ssl_evaluation_suite(
    ...,
    confidence_threshold: float = 0.70
):
```

3. **Passed to FixMatch trainer** (line 593):
```python
metrics_ssl = train_fixmatch_method(
    ...,
    confidence_threshold=confidence_threshold,  # Was: 0.95 hardcoded
    ...
)
```

4. **Connected in main()** (line 764):
```python
run_ssl_evaluation_suite(
    ...,
    confidence_threshold=args.confidence_threshold
)
```

### New Behavior:
```python
# NEW (train_ssl_evaluation_fixed.py:280)
adaptive_threshold = max(confidence_threshold, supervised_precision)
                    = max(0.70, 0.5689)
                    = 0.70  # BETTER!
```

**Expected Effect:**
- More pixels get pseudo-labels (threshold lowered from 95% to 70%)
- Better balance between precision and recall
- Recall should improve from 10% → 40-60%
- Overall F1 should improve to 60-70% range

---

## How to Run (Updated Command)

### Quick Test (2-3 hours):
```bash
python3 train_ssl_evaluation_fixed.py \
    --image-dir data/cryotransformer/images \
    --coords-file data/cryotransformer/coordinates.json \
    --output-dir experiments/ssl_test_threshold_070 \
    --labeled-ratios 0.05,0.10 \
    --target-size 768 \
    --particle-radius 42 \
    --batch-size 2 \
    --confidence-threshold 0.70
```

### Parameter Explanation:
```
--target-size 768            # Resize images to 768×768 (particles visible)
--particle-radius 42         # Correct radius from CSV (Diameter=84)
--batch-size 2               # Avoid OOM with 768×768 images
--confidence-threshold 0.70  # NEW! Lowered from 0.95 to allow more pseudo-labels
```

---

## Expected Results

### Before Fix (threshold=0.95):
```
5%  labels: Supervised F1 = 56.21%, FixMatch F1 = 42.24% (-14.0%)
10% labels: Supervised F1 = 56.86%, FixMatch F1 = 18.14% (-38.7%)

Issue: FixMatch making things WORSE!
```

### After Fix (threshold=0.70):
```
5%  labels: Supervised F1 = 56.21%, FixMatch F1 = ~62-65% (+6-9%)
10% labels: Supervised F1 = 56.86%, FixMatch F1 = ~63-67% (+7-10%)

Expected: FixMatch should HELP, not hurt!
```

---

## If Results Still Poor

If threshold=0.70 doesn't work well, you can tune it:

### Try Lower Threshold:
```bash
# More aggressive pseudo-labeling
python3 train_ssl_evaluation_fixed.py \
    ... (other args) \
    --confidence-threshold 0.60
```

### Try Higher Threshold:
```bash
# More conservative pseudo-labeling
python3 train_ssl_evaluation_fixed.py \
    ... (other args) \
    --confidence-threshold 0.80
```

### Optimal Range:
- **0.50-0.60**: Very aggressive (high recall, lower precision)
- **0.60-0.70**: Balanced (recommended)
- **0.70-0.80**: Conservative (higher precision, lower recall)
- **0.80-0.95**: Very conservative (may cause recall collapse)

---

## Monitoring Progress

### Check GPU usage:
```bash
watch -n 2 nvidia-smi
```

### Watch output directory:
```bash
watch -n 10 'ls -lh experiments/ssl_test_threshold_070/'
```

### View results as they complete:
```bash
# After 5% run completes
cat experiments/ssl_test_threshold_070/ssl_eval_fixmatch_ratio_5.json

# After 10% run completes
cat experiments/ssl_test_threshold_070/ssl_eval_fixmatch_ratio_10.json

# Final summary
cat experiments/ssl_test_threshold_070/ssl_evaluation_fixmatch_summary.json
```

---

## Summary

### What Was Wrong:
- Confidence threshold hardcoded at 0.95 (too high)
- Very few pseudo-labels generated
- Recall collapsed to 10%
- FixMatch degraded performance by -39%

### What Was Fixed:
- Added `--confidence-threshold` parameter
- Default changed to 0.70 (more reasonable)
- User can tune if needed
- Should improve recall significantly

### Next Step:
**Run the updated command above** and check if:
- ✅ Recall improves from 10% to 40-60%
- ✅ F1 improves from 18% to 60-70%
- ✅ FixMatch provides positive gain instead of degradation

Good luck! 🚀
