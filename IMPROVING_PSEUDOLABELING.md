# How to Improve Pseudo-Labeling Performance

**Current Status**: Pseudo-labeling provides 0% gain on 6K dataset

**Goal**: Find strategies to achieve +3-5% F1 improvement

---

## Why Current Approach Failed

### Problem 1: **Class Imbalance in Pseudo-Labels**

**Your dataset**:
- Ground truth: ~4% particles, 96% background
- Pseudo-labels: Likely ~1-2% particles, 98-99% background

**Why this happens**:
- Confidence = max(prob, 1-prob)
- Background at prob=0.10 → confidence=0.90 (KEPT)
- Particles at prob=0.60 → confidence=0.60 (FILTERED OUT)
- **Result**: Pseudo-labels have even fewer particles than ground truth

### Problem 2: **Too Many Pseudo-Labels**

**Current**:
- 258 labeled + 3,083 pseudo = **12:1 ratio**
- Pseudo-labels are ~85% accurate, real labels are 100% accurate
- **Result**: Noise overwhelms signal

### Problem 3: **Strong Baseline**

- Supervised already achieves 70% F1
- Hard to improve when baseline is this good
- Need very high-quality pseudo-labels

---

## 8 Strategies to Improve Performance

### ✅ **Strategy 1: Class-Balanced Selection** (MOST PROMISING)

**Idea**: Ensure pseudo-labels have same particle ratio as ground truth (4%)

**Implementation**:
```python
from improved_pseudolabel_strategies import generate_class_balanced_pseudo_labels

pseudo_images, pseudo_masks, stats = generate_class_balanced_pseudo_labels(
    model=model,
    unlabeled_images=unlabeled_images,
    device=device,
    target_particle_ratio=0.04,  # Match ground truth!
    confidence_threshold_particle=0.60,  # Lower for particles (harder)
    confidence_threshold_background=0.80,  # Higher for background (easier)
    max_pseudo_labels=258 * 2,  # At most 2× labeled count
    batch_size=4
)
```

**Expected improvement**: +2-5% F1

**Why it works**:
- Separate thresholds for particles (0.60) vs background (0.80)
- Selects images with balanced class distribution
- Avoids extreme background bias

---

### ✅ **Strategy 2: Limited Pseudo-Labels** (SIMPLEST)

**Idea**: Only add same amount of pseudo-labels as real labels (1:1 ratio)

**Current problem**:
- 258 real + 3,083 pseudo = 12:1 ratio (too diluted)

**Better approach**:
- 258 real + 258 pseudo = 1:1 ratio

**Implementation**:
```python
from improved_pseudolabel_strategies import generate_limited_pseudo_labels

pseudo_images, pseudo_masks, stats = generate_limited_pseudo_labels(
    model=model,
    unlabeled_images=unlabeled_images,
    labeled_count=258,  # Number of labeled images
    device=device,
    pseudo_to_labeled_ratio=1.0,  # 1:1 ratio
    confidence_threshold=0.70,
    batch_size=4
)
```

**Expected improvement**: +1-3% F1

**Why it works**:
- Prevents pseudo-label dilution
- Only keeps highest-confidence pseudo-labels
- Simple to implement and understand

---

### ✅ **Strategy 3: Curriculum Learning**

**Idea**: Start with very confident pseudo-labels, gradually add harder ones

**Progression**:
- Iteration 1: Threshold = 0.90 (only very confident)
- Iteration 2: Threshold = 0.75 (medium confidence)
- Iteration 3: Threshold = 0.60 (lower confidence)

**Implementation**:
```python
from improved_pseudolabel_strategies import generate_curriculum_pseudo_labels

for iteration in range(1, 4):
    pseudo_images, pseudo_masks, stats = generate_curriculum_pseudo_labels(
        model=model,
        unlabeled_images=unlabeled_images,
        device=device,
        iteration=iteration,
        max_iterations=3,
        initial_threshold=0.90,
        final_threshold=0.60,
        batch_size=4
    )
    # Train on labeled + pseudo
```

**Expected improvement**: +1-4% F1

**Why it works**:
- Model first learns from high-quality examples
- Gradually introduces more challenging examples
- Reduces risk of noise in early iterations

---

### ⚠️  **Strategy 4: Lower Confidence Threshold**

**Idea**: Use threshold 0.50 or 0.40 instead of 0.70

**Tradeoff**:
- ✅ Includes more particles (better recall of particle pseudo-labels)
- ❌ Includes more noise (lower precision)

**Test**:
```bash
python3 train_pseudolabel_coco.py \
    --train-data-dir data/cryotransformer_preprocessed/train \
    --val-data-dir data/cryotransformer_preprocessed/val \
    --output-dir experiments/pseudolabel_6k_threshold_050 \
    --labeled-ratios 0.05,0.10 \
    --confidence-threshold 0.50  # Changed from 0.70
```

**Expected improvement**: +0-2% F1 (risky - may hurt)

**When to use**: Only if class-balanced approach doesn't work

---

### 🔬 **Strategy 5: Weighted Loss for Pseudo-Labels**

**Idea**: Trust real labels more than pseudo-labels

**Implementation**:
```python
# In training loop:
for images, masks, is_pseudo in train_loader:
    outputs = model(images)

    # Calculate loss
    loss_all, _ = criterion(outputs, masks)

    # Weight real labels more than pseudo-labels
    weights = torch.where(is_pseudo, 0.5, 1.0)  # Pseudo gets 0.5 weight
    loss = (loss_all * weights).mean()

    loss.backward()
    optimizer.step()
```

**Expected improvement**: +1-2% F1

**Why it works**:
- Real labels guide training more strongly
- Pseudo-labels provide regularization without dominating

---

### 🔬 **Strategy 6: Ensemble Pseudo-Labels**

**Idea**: Use multiple models to generate pseudo-labels (more robust)

**Implementation**:
1. Train 3-5 models with different seeds
2. Generate pseudo-labels from each model
3. Only keep pseudo-labels where all models agree

**Expected improvement**: +2-4% F1 (but 3-5× more expensive)

**Why it works**:
- Reduces pseudo-label noise
- Only keeps high-consensus predictions

---

### 🔬 **Strategy 7: Mean Teacher Instead**

**Idea**: Use exponential moving average of weights instead of hard pseudo-labels

**Key difference**:
- Pseudo-labeling: Generate labels, then train on them
- Mean Teacher: Slowly update "teacher" model, use for consistency

**Implementation**: Would require new training script

**Expected improvement**: +1-3% F1

**Pros**:
- Smoother than pseudo-labeling
- Less sensitive to noise
- Well-suited for strong baselines

**Cons**:
- More complex to implement
- Requires new code

---

### 🔬 **Strategy 8: PU Learning Integration**

**Idea**: You already have PU learning framework - combine with pseudo-labeling!

**Approach**:
1. Use PU learning to identify reliable negatives (background)
2. Use pseudo-labeling for positives (particles)
3. Combine both for semi-supervised learning

**Implementation**: Modify existing PU code to work with unlabeled data

**Expected improvement**: +2-5% F1 (if PU works well)

**Why it works**:
- PU learning handles unlabeled negatives well
- Pseudo-labeling adds positive examples
- Best of both worlds

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)

**Try Strategy 1 (Class-Balanced) + Strategy 2 (Limited Quantity)**

```python
# Modified train_pseudolabel_coco.py to use:
from improved_pseudolabel_strategies import (
    generate_class_balanced_pseudo_labels,
    generate_limited_pseudo_labels
)

# Option A: Class-balanced (best for imbalanced data)
pseudo_images, pseudo_masks, stats = generate_class_balanced_pseudo_labels(
    model=current_model,
    unlabeled_images=unlabeled_images,
    device=device,
    target_particle_ratio=0.04,
    confidence_threshold_particle=0.60,
    confidence_threshold_background=0.80,
    max_pseudo_labels=len(labeled_idx) * 2,  # 2:1 ratio max
    batch_size=batch_size
)

# Option B: Limited quantity (simplest)
pseudo_images, pseudo_masks, stats = generate_limited_pseudo_labels(
    model=current_model,
    unlabeled_images=unlabeled_images,
    labeled_count=len(labeled_idx),
    device=device,
    pseudo_to_labeled_ratio=1.0,  # 1:1 ratio
    confidence_threshold=0.70,
    batch_size=batch_size
)
```

**Expected**: +2-5% F1 if class balance was the issue

---

### Phase 2: If Phase 1 Doesn't Work (3-5 days)

Try Strategy 3 (Curriculum) or Strategy 5 (Weighted Loss)

---

### Phase 3: Advanced (1 week)

Try Strategy 7 (Mean Teacher) or Strategy 8 (PU + Pseudo-labeling)

---

## Diagnostic: Check What's Wrong

Before trying new strategies, let's diagnose the problem:

```python
# Add this to your current pseudo-labeling code:
def diagnose_pseudo_labels(pseudo_masks, real_masks):
    """Check if pseudo-labels have class imbalance problem"""

    # Calculate particle ratios
    pseudo_particle_ratio = np.mean([m.mean() for m in pseudo_masks])
    real_particle_ratio = np.mean([m.mean() for m in real_masks])

    print(f"\n{'='*60}")
    print(f"PSEUDO-LABEL DIAGNOSIS")
    print(f"{'='*60}")
    print(f"Real labels particle ratio:   {real_particle_ratio*100:.2f}%")
    print(f"Pseudo-labels particle ratio: {pseudo_particle_ratio*100:.2f}%")
    print(f"Difference: {(pseudo_particle_ratio - real_particle_ratio)*100:+.2f}%")

    if pseudo_particle_ratio < real_particle_ratio * 0.5:
        print(f"\n⚠️  PROBLEM DETECTED: Pseudo-labels have too few particles!")
        print(f"   → Try Strategy 1 (Class-Balanced Selection)")
    elif pseudo_particle_ratio > real_particle_ratio * 2.0:
        print(f"\n⚠️  PROBLEM DETECTED: Pseudo-labels have too many particles!")
        print(f"   → Increase confidence threshold")
    else:
        print(f"\n✓ Pseudo-label class balance looks reasonable")

    print(f"{'='*60}\n")

# Run this after generating pseudo-labels
diagnose_pseudo_labels(pseudo_masks, train_masks[:len(labeled_idx)])
```

---

## Quick Test: Limited Pseudo-Labels

**Fastest test** - modify your current run to use limited quantity:

```bash
# Create new training script with limited pseudo-labels
cp train_pseudolabel_coco.py train_pseudolabel_coco_limited.py

# Edit: Change generate_pseudo_labels() to generate_limited_pseudo_labels()
# Then run:

python3 train_pseudolabel_coco_limited.py \
    --train-data-dir data/cryotransformer_preprocessed/train \
    --val-data-dir data/cryotransformer_preprocessed/val \
    --output-dir experiments/pseudolabel_6k_limited \
    --labeled-ratios 0.05,0.10 \
    --batch-size 4 \
    --confidence-threshold 0.70 \
    --n-iterations 3
```

**Expected runtime**: ~6-8 hours

**Expected result**:
- If +2-3% F1: Class balance was the problem!
- If +0-1% F1: Problem is elsewhere, try other strategies

---

## Summary: Best Strategies to Try

### **Ranked by Ease × Impact**:

1. **Limited Pseudo-Labels (Strategy 2)** - Easiest, likely +1-3% F1
2. **Class-Balanced Selection (Strategy 1)** - Easy, likely +2-5% F1
3. **Curriculum Learning (Strategy 3)** - Medium, likely +1-4% F1
4. **Weighted Loss (Strategy 5)** - Medium, likely +1-2% F1
5. **Mean Teacher (Strategy 7)** - Hard, likely +1-3% F1
6. **PU + Pseudo (Strategy 8)** - Hard, likely +2-5% F1

### **My Recommendation**:

Try **Strategy 1 (Class-Balanced)** first. It directly addresses the class imbalance problem that's likely causing the failure.

If that doesn't work, try **Strategy 2 (Limited Quantity)** - it's the simplest and prevents dilution.

---

## Reality Check

Even with these improvements, you might only get **+2-5% F1 gain** because:

1. Your supervised baseline is already strong (70% F1)
2. You have 258 labeled examples (not extremely scarce)
3. The task might be near its ceiling (~75% F1)

**Alternative**: Just report your excellent supervised results (70% F1 with 5% labels) and move on. This is already a strong contribution!
