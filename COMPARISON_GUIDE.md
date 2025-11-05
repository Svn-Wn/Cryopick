# FixMatch ON vs OFF Controlled Comparison Guide

This guide provides complete instructions for running a controlled comparison between Selective FixMatch (semi-supervised) and a supervised baseline.

---

## Quick Start (Copy-Paste Commands)

### 1. Run Baseline Training (FixMatch OFF)

```bash
# Train baseline model (supervised only, no FixMatch)
nohup python train_selective_fixmatch_baseline.py \
  --config configs/selective_fixmatch_baseline.yaml \
  --exp-name sfm_baseline_no_fixmatch \
  --device cuda:0 \
  > baseline_training.log 2>&1 &

# Monitor progress
tail -f baseline_training.log
```

**Expected time:** ~8-9 hours (50 epochs)

---

### 2. Run FixMatch Training (ON)

```bash
# Train full Selective FixMatch model (semi-supervised)
nohup python train_selective_fixmatch.py \
  --config configs/selective_fixmatch_v2_fast.yaml \
  --exp-name sfm_with_fixmatch \
  --device cuda:0 \
  > fixmatch_training.log 2>&1 &

# Monitor progress
tail -f fixmatch_training.log
```

**Expected time:** ~8-9 hours (50 epochs)

---

### 3. Generate Comparison Analysis

```bash
# Wait for both trainings to complete, then run analysis
python analyze_fixmatch_comparison.py
```

**Output:**
- Markdown report: `results_for_paper/comparison_summary.md`
- Plots: `results_for_paper/figures/`
- CSV metrics: `results_for_paper/csv/`

---

## Detailed Workflow

### Phase 1: Baseline Training (FixMatch OFF)

**What it does:**
- Trains a ResNet18 using **only supervised loss** on positive samples
- Uses **only weak augmentation** (no strong augmentation)
- Applies **class prior regularization** to prevent collapse
- **No consistency loss** (consistency_weight = 0.0)

**Config:** `configs/selective_fixmatch_baseline.yaml`

```yaml
loss:
  consistency_weight: 0.0  # DISABLED
  class_prior: 0.5
  prior_weight: 5.0
```

**Training command:**
```bash
python train_selective_fixmatch_baseline.py \
  --config configs/selective_fixmatch_baseline.yaml \
  --exp-name sfm_baseline_no_fixmatch \
  --device cuda:0
```

**Expected metrics:**
- Training loss: Supervised BCE + Prior penalty
- Mask rate: 0% (no pseudo-labeling)
- Val AUC: Baseline performance

**Output directory:** `experiments/sfm_baseline_no_fixmatch/`

---

### Phase 2: FixMatch Training (ON)

**What it does:**
- Trains ResNet18 with **supervised loss + consistency loss**
- Uses **weak + strong augmentation** for unlabeled samples
- Applies **pseudo-labeling** on high-confidence predictions
- Uses **class prior regularization** to prevent collapse

**Config:** `configs/selective_fixmatch_v2_fast.yaml`

```yaml
loss:
  consistency_weight: 1.0  # ENABLED
  confidence_threshold: 0.95
  class_prior: 0.5
  prior_weight: 5.0
```

**Training command:**
```bash
python train_selective_fixmatch.py \
  --config configs/selective_fixmatch_v2_fast.yaml \
  --exp-name sfm_with_fixmatch \
  --device cuda:0
```

**Expected metrics:**
- Training loss: Supervised BCE + Consistency + Prior
- Mask rate: 20-60% (high-confidence pseudo-labels used)
- Val AUC: Should improve over baseline if FixMatch helps

**Output directory:** `experiments/sfm_with_fixmatch/`

---

### Phase 3: Comparison Analysis

**What it does:**
- Loads `results.json` from both experiments
- Computes ΔAUC, ΔF1, and other metric differences
- Generates publication-quality comparison plots
- Writes a comprehensive markdown report

**Analysis command:**
```bash
python analyze_fixmatch_comparison.py
```

**Outputs:**

**1. Summary Report:** `results_for_paper/comparison_summary.md`
- Executive summary with winner declaration
- Metric comparison table (AUC, Precision, Recall, F1)
- % improvements and interpretation
- Recommendations

**2. Figures:** `results_for_paper/figures/`
- `roc_curves/roc_comparison.png` - ROC curve overlay
- `pr_curves/pr_comparison.png` - Precision-Recall scatter
- `f1_vs_threshold/f1_threshold_comparison.png` - Threshold sensitivity
- `training_dynamics/training_comparison.png` - Loss/AUC/Mask rate over epochs

**3. CSV Metrics:** `results_for_paper/csv/`
- `metrics_fixmatch_on.csv` - FixMatch test results
- `metrics_fixmatch_off.csv` - Baseline test results

---

## Expected Results

### Scenario 1: FixMatch Wins (ΔAUC > 0.01)

```
Baseline AUC: 0.8750
FixMatch AUC: 0.9150
ΔAUC: +0.0400 (+4.57%)

✅ FixMatch provides significant improvement!
```

**Interpretation:**
- Semi-supervised learning successfully leverages unlabeled data
- Consistency regularization improves generalization
- **Recommendation:** Use FixMatch for production

---

### Scenario 2: Marginal Improvement (0 < ΔAUC ≤ 0.01)

```
Baseline AUC: 0.8750
FixMatch AUC: 0.8800
ΔAUC: +0.0050 (+0.57%)

⚠️ FixMatch provides marginal improvement
```

**Interpretation:**
- Limited benefit from semi-supervised learning
- Supervised signal + prior regularization is strong
- **Recommendation:** Consider baseline for simplicity

---

### Scenario 3: Baseline Wins (ΔAUC < 0)

```
Baseline AUC: 0.8750
FixMatch AUC: 0.8650
ΔAUC: -0.0100 (-1.14%)

❌ Baseline outperforms FixMatch
```

**Interpretation:**
- Consistency loss adds noise rather than signal
- Possible issues: weak pseudo-labels, hyperparameters
- **Recommendation:** Investigate and tune FixMatch params

---

## Hyperparameter Differences

| Parameter | Baseline | FixMatch | Notes |
|-----------|----------|----------|-------|
| **consistency_weight** | 0.0 | 1.0 | Key difference |
| **augmentation** | weak only | weak + strong | FixMatch uses strong for consistency |
| **confidence_threshold** | N/A | 0.95 | Filters pseudo-labels |
| **class_prior** | 0.5 | 0.5 | Both use prior regularization |
| **prior_weight** | 5.0 | 5.0 | Prevents all-positive collapse |
| **lr** | 0.00003 | 0.00003 | Identical optimizer |
| **epochs** | 50 | 50 | Identical training length |

---

## Monitoring Training

### Check Training Progress

```bash
# Baseline
tail -f baseline_training.log | grep -E "Epoch|AUC"

# FixMatch
tail -f fixmatch_training.log | grep -E "Epoch|AUC|mask"
```

### Expected Logs

**Baseline (FixMatch OFF):**
```
Epoch 10 Results:
  Train Loss: 0.2450 | L_sup: 0.1950 | L_cons: 0.0000 | Mask: 0.00%
  Val AUC: 0.8523 | P: 0.8234 | R: 0.7891 | F1: 0.8059
```
*Note: L_cons=0, Mask=0% (no consistency loss)*

**FixMatch (ON):**
```
Epoch 10 Results:
  Train Loss: 0.2150 | L_sup: 0.1100 | L_cons: 0.0850 | Mask: 42.35%
  Val AUC: 0.8745 | P: 0.8456 | R: 0.8123 | F1: 0.8287
```
*Note: L_cons>0, Mask=40-60% (consistency loss active)*

---

## Troubleshooting

### Issue: Training collapses (pos_rate → 100%)

**Solution:** Prior regularization should prevent this. If it still happens:
```yaml
# Increase prior_weight in config
loss:
  prior_weight: 10.0  # Stronger regularization
```

---

### Issue: Mask rate stuck at 100%

**Problem:** Model too confident on all samples

**Solution:**
```yaml
# Increase confidence threshold
loss:
  confidence_threshold: 0.98  # More selective
```

---

### Issue: Both experiments give similar AUC

**Possible reasons:**
1. Dataset is small - semi-supervised learning needs more unlabeled data
2. Positive samples alone provide strong signal
3. Hyperparameters need tuning

**Next steps:**
- Try different confidence thresholds (0.90, 0.95, 0.98)
- Adjust consistency_weight (0.5, 1.0, 2.0)
- Use more training data if available

---

## Advanced: Grid Search for Best Threshold

The analysis script currently uses a fixed threshold (0.5). To find optimal threshold:

```python
# In analyze_fixmatch_comparison.py, add this function:

def grid_search_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1
```

*Note: This requires saving test predictions, which is not currently implemented*

---

## File Structure Summary

```
CryoEM_FixMatch_PU/
├── configs/
│   ├── selective_fixmatch_baseline.yaml      # Baseline config (consistency_weight=0)
│   └── selective_fixmatch_v2_fast.yaml       # FixMatch config (consistency_weight=1)
├── train_selective_fixmatch_baseline.py      # Baseline training script
├── train_selective_fixmatch.py               # FixMatch training script
├── analyze_fixmatch_comparison.py            # Comparison analysis
├── experiments/
│   ├── sfm_baseline_no_fixmatch/             # Baseline results
│   │   ├── results.json
│   │   ├── best_model.pt
│   │   └── best_model_ema.pt
│   └── sfm_with_fixmatch/                    # FixMatch results
│       ├── results.json
│       ├── best_model.pt
│       └── best_model_ema.pt
└── results_for_paper/                        # Comparison outputs
    ├── figures/
    │   ├── roc_curves/roc_comparison.png
    │   ├── pr_curves/pr_comparison.png
    │   ├── f1_vs_threshold/f1_threshold_comparison.png
    │   └── training_dynamics/training_comparison.png
    ├── csv/
    │   ├── metrics_fixmatch_on.csv
    │   └── metrics_fixmatch_off.csv
    └── comparison_summary.md
```

---

## Reproducibility Checklist

- [ ] Run baseline training to completion
- [ ] Run FixMatch training to completion
- [ ] Verify both `results.json` files exist
- [ ] Run comparison analysis
- [ ] Review `comparison_summary.md`
- [ ] Check all figures generated correctly
- [ ] Save CSV metrics for records

---

## Publication-Ready Outputs

The generated figures are 300 DPI and suitable for publication. Recommended usage:

1. **Figure 1 (Main Result):** `training_dynamics/training_comparison.png`
   - Shows validation AUC, F1, training loss, and mask rate
   - Demonstrates FixMatch benefit over training

2. **Figure 2 (ROC Comparison):** `roc_curves/roc_comparison.png`
   - Shows ΔAUC improvement
   - Key metric for binary classification

3. **Table 1 (Metrics):** Copy from `comparison_summary.md`
   - Test AUC, Precision, Recall, F1
   - % improvements

---

## Questions?

If you encounter issues or need modifications:
1. Check training logs for errors
2. Verify config files have correct parameters
3. Ensure both experiments completed successfully
4. Review `results.json` files for consistency

---

*Last updated: Generated from controlled comparison setup*
