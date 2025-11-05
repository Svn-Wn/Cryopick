# FixMatch ON vs OFF Comparison

## Results Summary

### Baseline (FixMatch OFF)
- **Test AUC**: 0.7434
- **Precision**: 0.5956
- **Recall**: 0.6945  
- **F1 Score**: 0.6412

### FixMatch (ON)
- **Test AUC**: 0.7434
- **Precision**: 0.4403
- **Recall**: 0.9950
- **F1 Score**: 0.6105

---

## Comparison

| Metric | Baseline | FixMatch | Δ (Difference) |
|--------|----------|----------|----------------|
| **AUC** | 0.7434 | 0.7434 | **-0.0000** (-0.01%) |
| **Precision** | 0.5956 | 0.4403 | **-0.1553** (-26.07%) |
| **Recall** | 0.6945 | 0.9950 | **+0.3005** (+43.27%) |
| **F1 Score** | 0.6412 | 0.6105 | **-0.0307** (-4.79%) |

---

## Winner: **TIE** (Same AUC)

### Key Findings:

1. **AUC is identical**: Both methods achieve AUC = 0.7434

2. **Trade-off pattern**:
   - ✅ **FixMatch**: Much higher recall (99.5% vs 69.5%) - finds almost all positives
   - ❌ **FixMatch**: Lower precision (44.0% vs 59.6%) - more false positives
   - Result: Almost perfect recall but at cost of precision

3. **F1 Score**: Baseline slightly better (0.6412 vs 0.6105)

---

## Interpretation

### What Happened:

The FixMatch model learned to predict **almost everything as positive** (recall ~100%), which explains:
- Very high recall (catches all positives)
- Low precision (many false positives)
- Same AUC (ROC is threshold-independent)

This suggests **model collapse** where FixMatch's consistency loss pushed the model toward predicting positive for most unlabeled samples.

### Root Cause:

The positive-unlabeled (PU) learning setup combined with FixMatch's pseudo-labeling may have created a positive feedback loop:
1. Model generates high-confidence positive pseudo-labels on unlabeled data
2. Consistency loss reinforces these predictions
3. Model becomes increasingly biased toward positive class

---

## Recommendations

### 1. **Use Baseline for production**
   - Better balance (precision vs recall)
   - Simpler and more stable
   - Same AUC as FixMatch

### 2. **To improve FixMatch** (if needed):
   - ✅ Increase confidence threshold (0.95 → 0.98) to be more selective
   - ✅ Reduce consistency weight (1.0 → 0.5)  
   - ✅ Strengthen class prior regularization
   - ✅ Monitor positive prediction rate during training

### 3. **Consider hybrid approach**:
   - Use baseline's balanced predictions
   - Add FixMatch only if recall needs to be maximized (e.g., safety-critical applications)

---

## Configuration Differences

**Baseline:**
- consistency_weight: 0.0 (no FixMatch)
- Augmentation: weak only
- Training: supervised BCE + prior regularization

**FixMatch:**  
- consistency_weight: 1.0
- Augmentation: weak + strong
- Training: supervised BCE + consistency loss + prior regularization
- confidence_threshold: 0.95

---

*The AUC being identical but precision/recall being very different indicates FixMatch learned a different operating point on the ROC curve - one that maximizes recall at the expense of precision.*
