# Traditional Pseudo-Labeling for Cryo-EM

**Status:** Running now (background process started)
**Date:** October 15, 2025

---

## Why Pseudo-Labeling Instead of FixMatch?

### FixMatch Failed Because:
```
Results with FixMatch:
  5%  labels: F1 = 1.57%  (Recall = 0.80%)  ❌
  10% labels: F1 = 18.15% (Recall = 10.35%) ❌

Problem: Consistency loss on strong augmentations amplified class imbalance
→ Model learned to predict background everywhere
→ Recall collapsed to <1%
```

### Pseudo-Labeling Is Better For Imbalanced Data:

**FixMatch approach:**
1. Generate pseudo-labels from weak augmentation
2. Train on consistency between weak → strong augmentation
3. **Problem**: Consistency loss amplifies bias toward majority class (background)

**Pseudo-Labeling approach:**
1. Generate pseudo-labels from model predictions
2. Simply add them to training data
3. Retrain on combined labeled + pseudo-labeled data
4. **Advantage**: No consistency loss, just normal supervised training

---

## How It Works

### Algorithm:

```
1. Train supervised model on labeled data
   → Get baseline performance (e.g., F1 = 56%)

2. Use model to predict on unlabeled data
   → Generate pseudo-labels for unlabeled images
   → Filter by confidence (keep images with avg confidence > 0.70)

3. Combine labeled + pseudo-labeled data
   → Original: 45 labeled images
   → Pseudo:   X unlabeled images (high confidence)
   → Total:    45 + X images

4. Retrain model on combined data
   → Train for 50 epochs

5. If improved, iterate (repeat steps 2-4)
   → Up to 3 iterations
   → Stop if no improvement

6. Return best model
```

### Key Differences from FixMatch:

| Aspect | FixMatch | Pseudo-Labeling |
|--------|----------|-----------------|
| **Augmentation** | Weak + Strong | Normal training augmentation |
| **Loss** | Supervised + Consistency | Just supervised |
| **Pseudo-labels** | Used for consistency target | Added to training data |
| **Class imbalance** | Amplified by consistency loss | Handled normally |
| **Iterations** | Single training run | Multiple iterations (self-training) |

---

## Current Run Parameters

```bash
python3 train_pseudolabel.py \
    --image-dir data/cryotransformer/images \
    --coords-file data/cryotransformer/coordinates.json \
    --output-dir experiments/pseudolabel_test \
    --labeled-ratios 0.05,0.10 \
    --target-size 768 \
    --particle-radius 42 \
    --batch-size 2 \
    --confidence-threshold 0.70 \
    --n-iterations 3
```

### Parameters Explained:

- `--confidence-threshold 0.70`: Only keep unlabeled images where average confidence > 70%
- `--n-iterations 3`: Up to 3 rounds of pseudo-labeling
- Other parameters: Same as supervised baseline

---

## Expected Results

### At 5% Labels (45 labeled, 855 unlabeled):

**Iteration 1:**
```
1. Supervised baseline: F1 = ~56%
2. Generate pseudo-labels on 855 unlabeled
   → Keep ~400-600 high-confidence images
3. Retrain on 45 labeled + 400 pseudo
   → Expected: F1 = ~58-62% (+2-6% improvement)
```

**Iteration 2:**
```
1. Use improved model to generate new pseudo-labels
   → Keep ~500-700 high-confidence images
2. Retrain on 45 labeled + 500-700 pseudo
   → Expected: F1 = ~60-65% (if improving)
```

**Iteration 3:**
```
1. Continue if still improving
2. Otherwise stop and return best model
```

### At 10% Labels (90 labeled, 810 unlabeled):

Similar process, but:
- Baseline already ~57% F1
- May see smaller gains (0-3%)
- May hit diminishing returns faster

---

## Why This Should Work

### 1. No Consistency Loss
- **FixMatch problem**: Consistency loss forced model to match predictions on strong augmentations
- **Our solution**: Just add pseudo-labels to training set, train normally
- **Result**: No amplification of class imbalance

### 2. Confidence Filtering
- **Threshold 0.70**: Only keep images where model is confident (on average)
- **Effect**: Filter out ambiguous/uncertain predictions
- **Advantage**: Adds high-quality pseudo-labels

### 3. Iterative Refinement
- **Round 1**: Add pseudo-labels from baseline model
- **Round 2**: Improved model generates better pseudo-labels
- **Round 3**: Further refinement if beneficial
- **Stop**: When no longer improving

### 4. Handles Imbalance Better
- Normal supervised loss with pos_weight=5.0
- No special consistency loss that favors majority class
- Model trains on balanced objective (Focal + Dice loss)

---

## Timeline

### Current Run:
- **Started**: October 15, 2025
- **Status**: Running in background
- **Duration**: ~3-4 hours for both 5% and 10% labels

### Per Label Ratio:
```
5% labels:
  - Supervised baseline: ~1 hour
  - Iteration 1: ~40 min (generate pseudo + retrain)
  - Iteration 2: ~40 min (if improves)
  - Iteration 3: ~40 min (if improves)
  - Total: ~2-3 hours

10% labels:
  - Similar timing
  - May terminate early if no improvement
```

---

## Monitoring Progress

### Check progress:
```bash
# Watch output directory
watch -n 10 'ls -lh experiments/pseudolabel_test/'

# Check GPU
nvidia-smi

# View logs (when available)
tail -f nohup.out  # If running in background
```

### Expected files:
```
experiments/pseudolabel_test/
├── pseudolabel_ratio_5.json    # 5% results
├── pseudolabel_ratio_10.json   # 10% results
└── pseudolabel_summary.json    # Overall summary
```

---

## What to Expect

### Success Criteria:
```
✅ Recall should NOT collapse (stay > 50%)
✅ F1 should improve by 2-5% at 5% labels
✅ No degradation at 10% labels
✅ Pseudo-labeling should stop when not beneficial
```

### Likely Outcomes:

**Best case:**
```
5%  labels: Supervised F1 = 56% → Pseudo F1 = 62-65% (+6-9%)
10% labels: Supervised F1 = 57% → Pseudo F1 = 59-61% (+2-4%)
```

**Realistic:**
```
5%  labels: Supervised F1 = 56% → Pseudo F1 = 58-60% (+2-4%)
10% labels: Supervised F1 = 57% → Pseudo F1 = 58-59% (+1-2%)
```

**Worst case (still better than FixMatch!):**
```
5%  labels: Supervised F1 = 56% → Pseudo F1 = 56-57% (+0-1%)
10% labels: Supervised F1 = 57% → Pseudo F1 = 57-58% (+0-1%)
```

**Key**: Even if gains are small, recall won't collapse!

---

## Comparison with FixMatch

| Metric | FixMatch (Failed) | Pseudo-Labeling (Expected) |
|--------|-------------------|----------------------------|
| **5% F1** | 1.57% | 58-60% |
| **5% Recall** | 0.80% | 55-60% |
| **10% F1** | 18.15% | 58-60% |
| **10% Recall** | 10.35% | 55-60% |
| **Gain at 5%** | -54.64% | +2-4% |
| **Gain at 10%** | -38.71% | +1-2% |

**Key improvement:** Recall stays healthy, no catastrophic collapse!

---

## If Results Are Still Poor

If pseudo-labeling also doesn't help, we have fallback options:

### Option 1: Just Use Supervised Learning
- Your supervised baseline already gets 56-57% F1
- Sometimes SSL doesn't help on small/imbalanced datasets
- **Honest positioning in paper**: "SSL provides minimal benefit on this dataset"

### Option 2: Try Different SSL Methods
- **Mean Teacher**: Exponential moving average of model weights
- **MixMatch**: Combines mixup with pseudo-labeling
- **DARP**: Designed specifically for imbalanced data

### Option 3: More Data Augmentation
- Focus on improving supervised baseline
- Better augmentations → Better baseline → Less need for SSL

---

## Summary

**What we're doing now:**
- Traditional pseudo-labeling (self-training)
- Simple, proven approach
- Better suited for imbalanced data than FixMatch

**Why it should work:**
- No consistency loss to amplify bias
- Confidence filtering keeps quality high
- Iterative refinement improves over time
- Natural stopping when not beneficial

**Expected outcome:**
- Small but positive gains (2-4% at 5% labels)
- No recall collapse
- Better than supervised baseline
- Much better than FixMatch!

**Wait time:** ~3-4 hours for complete results

Let the training run and check results when done! 🚀
