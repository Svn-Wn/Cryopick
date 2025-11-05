# Class-Balanced Pseudo-Labeling Experiment

**Status**: ✅ Running (Process ID: fab089)
**Started**: Just now
**Expected Duration**: ~15-20 hours for complete experiment

---

## What's Different from Previous Run?

### Previous (Naive Approach) - FAILED:

```python
# Single confidence threshold for all pixels
confidence = max(prob, 1-prob)
keep_if confidence >= 0.70

Result:
- 3,083 pseudo-labels at 5% (12:1 ratio)
- Pseudo-labels had ~2% particles (too few!)
- Performance: 70.00% → 69.85% F1 (degraded)
- Gain: 0%
```

---

### Current (Class-Balanced) - EXPECTED TO WORK:

```python
# SEPARATE thresholds for particles vs background
particle_threshold = 0.60  # Lower (easier to pass)
background_threshold = 0.80  # Higher (harder to pass)

# Select images with balanced class distribution
target_particle_ratio = 4%  # Match ground truth!
max_pseudo_labels = 258 × 2 = 516  # Limit to 2:1 ratio

Expected Result:
- ~500-1,000 pseudo-labels at 5% (2-4:1 ratio)
- Pseudo-labels have ~4% particles (balanced!)
- Performance: 70.00% → 73-75% F1 (improved!)
- Expected gain: +3-5%
```

---

## Key Improvements

### 1. **Separate Confidence Thresholds**

| Class | Naive Approach | Class-Balanced |
|-------|----------------|----------------|
| **Particles** (hard class) | Threshold 0.70 | Threshold **0.60** (lower) |
| **Background** (easy class) | Threshold 0.70 | Threshold **0.80** (higher) |

**Effect**: Keeps more particles, filters more background → Balanced!

---

### 2. **Quality Score Based on Class Balance**

```python
# OLD: Just use average confidence
quality = avg_confidence

# NEW: Heavily weight class balance
quality = (
    0.3 × particle_confidence +
    0.2 × background_confidence +
    0.5 × balance_score  # 50% weight on having 4% particles!
)
```

**Effect**: Prefers images with particle ratio close to 4%

---

### 3. **Limited Quantity**

```python
# OLD: Keep all that pass threshold
# Result: 3,083 pseudo-labels (12:1 ratio)

# NEW: Limit to 2× labeled count
max_pseudo = 258 × 2 = 516
# Result: ~500 highest-quality pseudo-labels (2:1 ratio)
```

**Effect**: Prevents dilution from noisy pseudo-labels

---

## Expected Timeline

### Data Loading (Current Phase):
```
⏳ Loading 5,172 training images (~2 minutes)
⏳ Loading 534 validation images (~20 seconds)
```

### 5% Labeled (258 images):
```
1. Supervised baseline (100 epochs)            ~3-4 hours
2. Generate class-balanced pseudo-labels       ~5 minutes
   → Expected: ~500-1,000 pseudo-labels (2-4:1 ratio)
   → Particle ratio: ~4% (balanced!)

3. Train on 258 labeled + 500 pseudo (50 epochs)  ~2 hours
   → Expected: 70.00% → 72-73% F1 (+2-3%)

4. Iteration 2 (if improved)                   ~2 hours
   → Expected: 73% → 74% F1 (+1%)

5. Iteration 3 (if improved)                   ~2 hours
   → Expected: 74% → 74-75% F1 (+0-1%)

Total for 5%: ~8-12 hours
```

### 10% Labeled (517 images):
```
Similar process
Total: ~6-10 hours
```

**Overall ETA: 15-20 hours**

---

## How to Monitor Progress

### Check Log File:
```bash
# View latest output
tail -100 pseudolabel_6k_balanced_training.log

# Watch in real-time
tail -f pseudolabel_6k_balanced_training.log

# Check GPU usage
nvidia-smi
```

### Look for Key Messages:

```
✅ Generated X class-balanced pseudo-labels
   Average particle ratio: Y% (target: 4.0%)
```

**Good signs**:
- Y% close to 4% (e.g., 3.5-4.5%)
- X in range 500-1,000 at 5% labels
- Ratio to labeled: 2-4:1 (not 12:1 like before)

**Bad signs**:
- Y% < 2% or > 6% (still imbalanced)
- X > 2,000 (too many, likely diluted)
- No improvement in iterations

---

## Expected Results

### Best Case Scenario:

| Labeled | Naive (Previous) | Class-Balanced (Expected) | Improvement |
|---------|------------------|---------------------------|-------------|
| 5% (258) | 70.00% F1 | **73-75% F1** | **+3-5%** ✅ |
| 10% (517) | 70.98% F1 | **73-75% F1** | **+2-4%** ✅ |

---

### Realistic Scenario:

| Labeled | Naive (Previous) | Class-Balanced (Expected) | Improvement |
|---------|------------------|---------------------------|-------------|
| 5% (258) | 70.00% F1 | **72-73% F1** | **+2-3%** ✅ |
| 10% (517) | 70.98% F1 | **72-73% F1** | **+1-2%** ✅ |

---

### Worst Case (if class balance wasn't the issue):

| Labeled | Naive (Previous) | Class-Balanced (Expected) | Improvement |
|---------|------------------|---------------------------|-------------|
| 5% (258) | 70.00% F1 | **70-71% F1** | **+0-1%** ⚠️ |
| 10% (517) | 70.98% F1 | **71-72% F1** | **+0-1%** ⚠️ |

---

## Files and Locations

**Training script**: `train_pseudolabel_coco_balanced.py`

**Log file**: `pseudolabel_6k_balanced_training.log`

**Results directory**: `experiments/pseudolabel_6k_balanced/`

**Output files** (when complete):
```
experiments/pseudolabel_6k_balanced/
├── balanced_pseudolabel_ratio_5.json     # 5% results
├── balanced_pseudolabel_ratio_10.json    # 10% results
└── balanced_pseudolabel_summary.json     # Overall summary
```

---

## What Makes This Different?

### Comparison Table:

| Aspect | Naive (0% gain) | Class-Balanced (Expected: +3-5%) |
|--------|-----------------|----------------------------------|
| **Particle threshold** | 0.70 (too high) | **0.60** (lower for hard class) |
| **Background threshold** | 0.70 (too low) | **0.80** (higher for easy class) |
| **Pseudo-label selection** | All that pass | **Top K by quality score** |
| **Quality metric** | Avg confidence | **50% balance + 50% confidence** |
| **Particle ratio target** | None | **4% (match ground truth)** |
| **Max pseudo-labels** | Unlimited | **2× labeled count** |
| **Pseudo ratio (5%)** | 12:1 | **2:1** |
| **Pseudo particle %** | ~2% (too few) | **~4% (balanced!)** |

---

## Success Criteria

✅ **Success if**:
- Pseudo-labels have 3-5% particles (close to 4% target)
- F1 improves by +2% or more
- Recall doesn't degrade
- Uses 500-1,000 pseudo-labels (not 3,000+)

❌ **Failure if**:
- Pseudo-labels still have <2% particles (imbalance persists)
- F1 improves by <1%
- Performance degrades like before

⚠️ **Partial success if**:
- F1 improves by +1-2% (better than nothing, but not ideal)

---

## Next Steps After This Run

### If Successful (+3-5% gain):
1. ✅ Write paper reporting SSL results
2. Run multiple seeds for statistical significance
3. Compare with CryoTransformer (optional)
4. Celebrate! 🎉

### If Partial Success (+1-2% gain):
1. Try other strategies (curriculum learning, weighted loss)
2. Consider reporting as "modest SSL benefit"
3. Still better than naive approach

### If Still Fails (0% gain):
1. Report excellent supervised results (70% F1 with 5% labels)
2. Include SSL as negative result in ablation
3. Write paper on "when SSL doesn't help"

---

## Monitor in Real-Time

Right now, the experiment is:
```
✅ Running (Process fab089)
⏳ Loading data (batch 8/52)
⏳ Next: Calculate ground truth particle ratio
⏳ Then: Start supervised baseline training
```

Check status:
```bash
tail -50 pseudolabel_6k_balanced_training.log
```

---

**Bottom Line**: This addresses the root cause of why naive pseudo-labeling failed (class imbalance). Expected to achieve **+2-5% F1 improvement**!
