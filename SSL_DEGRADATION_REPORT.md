# SSL Performance Degradation Analysis

**Date:** October 14, 2025
**Issue:** SSL underperforms supervised learning at 10%+ label ratios
**Status:** ROOT CAUSES IDENTIFIED

---

## Executive Summary

Your SSL evaluation shows **SSL only helps at 5% labels (+2.1% F1) but degrades performance at higher ratios**:

| Label Ratio | Supervised F1 | SSL F1 | Gain |
|------------|---------------|---------|------|
| 5% | 92.2% | 94.3% | **+2.1%** ✓ |
| 10% | 96.6% | 95.7% | **-0.9%** ✗ |
| 20% | 97.2% | 96.6% | **-0.7%** ✗ |
| 50% | 97.1% | 95.2% | **-2.0%** ✗ |

**Critical Finding:** Your implementation is **NOT FixMatch** - it's basic self-training with pseudo-labels.

---

## Root Causes

### 1. CONFIRMATION BIAS (Primary Issue)

**Problem:** Model retrains on its own predictions → amplifies biases

**Evidence:**
- At 50% labels: SSL loses -2.2% precision AND -1.8% recall (both degrade)
- Self-training creates "echo chamber" effect
- Model learns to fit its own confident mistakes

**Why it happens:**
```
Supervised model → Generate pseudo-labels → Retrain on pseudo-labels
       ↑                                              ↓
       └──────────────── Confirmation loop ──────────┘
```

At high label ratios:
- Base model is already strong (97% precision)
- Pseudo-labels capture model's systematic errors
- Retraining reinforces these errors instead of correcting them

---

### 2. NO EARLY STOPPING MECHANISM

**Problem:** SSL always runs 2 iterations regardless of whether it helps

**Current behavior:**
```python
# From train_ssl_evaluation.py:434
if labeled_ratio < 1.0:
    # Use SSL only if we have unlabeled data
    metrics_ssl = train_ssl_method(...)
```

**Missing:**
- No validation-based stopping criterion
- No check if SSL is improving over supervised
- Should stop after iteration 0 (supervised-only) when base model is strong

**Proposed fix:**
```python
if labeled_ratio < 1.0 and supervised_f1 < 0.95:
    # Only use SSL if supervised baseline is weak
    metrics_ssl = train_ssl_method(...)
else:
    # Skip SSL if supervised is already strong
    metrics_ssl = metrics_sup
```

---

### 3. HARD PSEUDO-LABELS WITHOUT CONFIDENCE WEIGHTING

**Problem:** Treats all pseudo-labels equally, regardless of confidence

**Current approach:**
```python
# From train_ssl_evaluation.py:276-278
pseudo = torch.zeros_like(probs)
pseudo[probs >= pos_thresh] = 1.0  # Hard label
pseudo[probs <= neg_thresh] = 0.0  # Hard label
```

**Issues:**
- Prediction at 0.951 confidence → treated same as 0.999 confidence
- At high label ratios, captures confident mistakes (e.g., model systematically misclassifies certain patterns)
- No downweighting for uncertain predictions

**Proposed fix:**
```python
# Soft pseudo-labels with confidence weighting
pseudo_labels = probs.clone()  # Keep probabilities
pseudo_confidence = torch.abs(probs - 0.5) * 2  # 0 to 1

# In loss computation
loss_ssl = pseudo_confidence * criterion(outputs, pseudo_labels)
```

---

### 4. MISSING FIXMATCH IMPLEMENTATION

**CRITICAL:** Your SSL evaluation script does NOT implement FixMatch!

**What's missing:**

```python
# FixMatch requires:
# 1. Weak augmentation (for pseudo-label generation)
weak_aug = WeakAugmentation()  # Only flip + translate
weak_pred = model(weak_aug(unlabeled_image))
pseudo_label = (weak_pred > threshold).float()

# 2. Strong augmentation (for consistency)
strong_aug = StrongAugmentation()  # + RandAugment, cutout
strong_pred = model(strong_aug(unlabeled_image))

# 3. Consistency loss (MISSING!)
consistency_loss = criterion(strong_pred, pseudo_label)
```

**Current implementation:** Basic self-training
- Generate pseudo-labels once
- Retrain on combined dataset
- No consistency regularization
- No weak/strong augmentation contrast

**This is a major discrepancy** - you claimed "ResNet-18 + PU + FixMatch" but implemented self-training.

---

### 5. DISTRIBUTION SHIFT

**Problem:** Pseudo-labeled data != true data distribution

At 5% labels:
- Supervised: Underfits due to lack of data
- SSL pseudo-labels: Provide additional training signal
- Net effect: **Positive (+2.1% F1)**

At 50% labels:
- Supervised: Already represents true distribution well
- SSL pseudo-labels: Biased toward model's learned patterns
- Mixing original + pseudo → dilutes true distribution
- Net effect: **Negative (-2.0% F1)**

**Mathematical perspective:**
```
P_true(y|x)     = true particle distribution
P_model(y|x)    = model's learned distribution (biased)

At 5% labels:  P_model ≈ P_true + noise → pseudo-labels add signal
At 50% labels: P_model ≈ P_true + small_bias → pseudo-labels add bias
```

---

## Precision-Recall Analysis

### At 5% labels (SSL WORKS):
```
Supervised: Precision=85.7%, Recall=99.8%
→ Model overly recalls, low precision (guessing "particle" too often)

SSL:        Precision=89.9%, Recall=99.1%
→ High-confidence pseudo-labels filter false positives
→ Improves precision with minimal recall loss
→ NET: +2.1% F1 ✓
```

### At 50% labels (SSL FAILS):
```
Supervised: Precision=97.2%, Recall=97.1%
→ Model is already well-calibrated

SSL:        Precision=95.1%, Recall=95.3%
→ Pseudo-labels introduce noise in both directions
→ Both precision AND recall degrade
→ NET: -2.0% F1 ✗
```

**Key insight:** SSL hurts when supervised model is already well-calibrated.

---

## Proposed Fixes

### CRITICAL (Must Fix Before Publication):

#### 1. Add Validation-Based Early Stopping
```python
def should_use_ssl(supervised_metrics, labeled_ratio, threshold=0.95):
    """Only use SSL if supervised model is weak enough to benefit"""
    return supervised_metrics['f1_score'] < threshold and labeled_ratio < 0.50
```

#### 2. Implement True FixMatch
```python
# Weak augmentation for pseudo-label generation
weak_aug = Compose([RandomHorizontalFlip(), RandomTranslate()])
weak_output = model(weak_aug(unlabeled_images))
pseudo_labels = (weak_output > confidence_threshold).float()

# Strong augmentation for consistency
strong_aug = Compose([weak_aug, RandAugment(), Cutout()])
strong_output = model(strong_aug(unlabeled_images))

# Consistency loss (THIS IS THE KEY!)
mask = torch.max(weak_output, dim=1)[0] > confidence_threshold
loss_consistency = (mask * F.cross_entropy(strong_output, pseudo_labels, reduction='none')).mean()

# Total loss
loss = loss_supervised + lambda_u * loss_consistency
```

#### 3. Add Confidence Weighting
```python
# Instead of hard pseudo-labels
pseudo_confidence = torch.max(probs, dim=1)[0]
loss_mask = pseudo_confidence > adaptive_threshold

# Weight by confidence
loss_ssl = (loss_mask.float() * pseudo_confidence * criterion(...)).mean()
```

---

### IMPORTANT (Should Fix):

#### 4. Pseudo-Label Quality Filtering
```python
# Only use pseudo-labels with confidence > supervised model's performance
confidence_threshold = max(0.95, supervised_metrics['precision'])
pseudo_labels = probs > confidence_threshold
```

#### 5. Curriculum Pseudo-Labeling
```python
# Start with most confident, gradually relax
for iteration in range(num_iterations):
    threshold = max(0.90, 0.99 - iteration * 0.02)
    if val_f1_improves:
        continue
    else:
        break  # Stop if not improving
```

#### 6. Data Mixing Strategy
```python
# Preserve original distribution
alpha = min(0.5, 1 - labeled_ratio)  # More pseudo-labels at low label ratios
combined_data = alpha * pseudo_labeled + (1-alpha) * original_labeled
```

---

### NICE TO HAVE:

#### 7. Ensemble Pseudo-Labeling
```python
# Train multiple models or use model checkpoints
pseudo_labels = []
for model_snapshot in ensemble:
    pseudo_labels.append(model_snapshot(unlabeled_data))

# Only use predictions where ensemble agrees
agreement = torch.std(pseudo_labels, dim=0) < threshold
final_pseudo = torch.mean(pseudo_labels, dim=0)[agreement]
```

#### 8. Per-Class Threshold Adaptation (FlexMatch)
```python
# Adapt threshold based on class learning status
threshold_positive = compute_threshold(positive_learning_status)
threshold_negative = compute_threshold(negative_learning_status)
```

---

## Impact on Paper Readiness

### Current Status: **NOT READY FOR PUBLICATION**

**Major issues:**
1. ✗ Method mislabeled as "FixMatch" but is actually basic self-training
2. ✗ SSL degrades performance at 10%+ labels (red flag for reviewers)
3. ✗ No proper ablation of SSL components
4. ✗ No comparison with real FixMatch implementation

**Minimal viable fixes for publication:**
1. ✓ Implement true FixMatch with consistency regularization
2. ✓ Add early stopping to prevent degradation
3. ✓ Show SSL-only helps at <10% labels (be honest in paper)
4. ✓ Add ablation: Self-training vs FixMatch vs Supervised

**Recommended timeline:**
- 1 week: Implement true FixMatch
- 1 week: Re-run all experiments with fixed method
- 1 week: Ablation studies and comparisons
- **Total: ~3 weeks to be publication-ready**

---

## Key Takeaways

1. **Your SSL is NOT FixMatch** - it's basic self-training
2. **SSL only helps at 5% labels** because supervised model is weak there
3. **Confirmation bias is the main culprit** at higher ratios
4. **Must implement proper FixMatch** with consistency regularization
5. **Add early stopping** to prevent degradation
6. **Be honest in paper:** "SSL helps only in extreme low-data regimes (<10% labels)"

---

## Next Steps

1. Implement true FixMatch with weak/strong augmentation + consistency loss
2. Add validation-based early stopping
3. Re-run SSL evaluation suite
4. Compare: Supervised vs Self-Training vs FixMatch
5. If still degrades at high ratios, only claim SSL benefit at <10% labels

---

## References for Implementation

- **FixMatch paper:** Sohn et al. 2020, "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
- **Key equation:** L_total = L_supervised + λ_u × L_consistency
- **Consistency loss:** MSE(model(strong_aug(x)), pseudo_label_from_weak_aug(x))

Good luck! This is fixable but needs proper FixMatch implementation.
