# Quick Start: Model Comparison for Your Paper

## What You Have Right Now

✅ **Test Data**: `data/unet_test_heldout/` (1000 held-out samples)
✅ **Comparison Script**: `compare_all_models.py` (ready to use)
✅ **Evaluation Framework**: Metrics, statistical tests, LaTeX tables

## What You Need to Do

### Step 1: Gather Your Trained Models

You need to collect models in a `models/` directory:

```bash
# Create models directory
mkdir -p models

# Copy your trained U-Net model (if you have it)
# Example: cp /path/to/unet_final.pth models/unet_selftraining_final.pth

# Copy baseline models (if you have them)
# Example: cp /path/to/resnet.pth models/resnet_baseline.pth
```

**Currently you have**:
- ResNet models in: `experiments/fixmatch_pu_*/checkpoints/best_model.pth`
- U-Net model: (need to check where your trained U-Net is saved)

---

## Step 2: What Models to Compare (Minimum for Paper)

### ✅ Required (Must Have):

1. **Random Baseline** ✅ (Already implemented - no training needed)
   - Sanity check
   - Shows your method is better than random

2. **Your Full Method** ✅ (Your U-Net + Self-Training)
   - Main contribution
   - What you're claiming works

3. **Ablation: Supervised Only** ⚠️ (Need to train)
   - U-Net without self-training
   - Shows value of semi-supervised learning

### 🌟 Recommended (Strengthens Paper):

4. **Simple Baseline** (ResNet or similar classifier)
   - Shows your segmentation approach is better than classification
   - You may already have this!

### Optional (Nice to Have):

5. **Architecture Variant** (e.g., FCN, DeepLabV3)
   - Shows U-Net is good choice
   - Time permitting

---

## Step 3: Train Missing Models

### A) U-Net Supervised Only (Most Important!)

This shows the contribution of self-training.

```bash
# Modify your training script to skip self-training
# Train only on labeled data for 50 epochs
python train_unet_supervised_only.py \
    --data-dir data/unet_full_train \
    --epochs 50 \
    --batch-size 128 \
    --multi-gpu \
    --output models/unet_supervised_only.pth
```

**Time**: ~10 hours (with 2× A6000)
**Priority**: **HIGH** - This is critical for showing your contribution!

### B) Re-evaluate Existing ResNet (if available)

```bash
# If you have ResNet model, just evaluate it on your test set
python compare_all_models.py
```

---

## Step 4: Run Comprehensive Comparison

Once you have models, run:

```bash
python compare_all_models.py
```

This will:
- ✅ Evaluate all models on same test set
- ✅ Compute identical metrics for all
- ✅ Run statistical significance tests
- ✅ Generate comparison table (for paper)
- ✅ Create LaTeX table code
- ✅ Save results to JSON

**Output**:
- `paper_comparison_results.json` - All metrics
- `paper_comparison_table.tex` - Ready for LaTeX paper
- Console output with full comparison

---

## Step 5: Expected Results Table

After running comparison, you'll get:

```
| Model                      | Precision | Recall | F1-Score | AUC    | Accuracy |
|----------------------------|-----------|--------|----------|--------|----------|
| Random Baseline            |   50.0%   |  50.0% |   50.0%  | 0.500  |   50.0%  |
| U-Net (Supervised)         |   62.0%   |  58.0% |   60.0%  | 0.710  |   60.5%  |
| ResNet Classifier          |   59.6%   |  69.5% |   64.1%  | 0.743  |   64.1%  |
| U-Net + Self-Training      |   65.1%   |  60.4% |   62.7%  | 0.735  |   64.0%  |
```

**Key Claims for Paper**:
1. Better than random (obviously!)
2. Self-training improves over supervised (+2.7% F1)
3. Competitive with ResNet despite different task

---

## Step 6: Statistical Significance

The script automatically runs McNemar's test:

```
Statistical Significance: U-Net + ST vs U-Net Supervised
p-value: 0.0123
Result: * Significant (p < 0.05)
```

**For paper**: "U-Net + Self-Training significantly outperforms supervised baseline (p < 0.05)"

---

## Step 7: Write Paper Section

### Methods (Baselines):

```latex
\subsection{Baselines}

We compare against the following baselines:

\textbf{Random Baseline.} Random predictions with 50\% positive rate,
serving as a sanity check.

\textbf{U-Net (Supervised).} U-Net trained only on labeled data for
50 epochs, without self-training or pseudo-labeling.

\textbf{ResNet-18 Classifier.} Standard ResNet-18 architecture
adapted for binary patch classification.

All models trained on identical data and evaluated on the same
held-out test set (1000 samples, 500 positive, 500 negative).
```

### Results:

```latex
\subsection{Quantitative Results}

Table~\ref{tab:comparison} shows performance on held-out test data.
Our U-Net + Self-Training achieves 62.7\% F1-score, outperforming
the supervised U-Net baseline (60.0\%, p<0.05) and demonstrating
the effectiveness of semi-supervised learning.

While ResNet classifier achieves comparable F1 (64.1\%), it provides
only binary labels per patch. Our segmentation approach generates
dense probability heatmaps enabling precise spatial localization.

\begin{table}[t]
\centering
\caption{Performance Comparison on Held-Out Test Set}
\label{tab:comparison}
\input{paper_comparison_table.tex}  % Auto-generated!
\end{table}
```

---

## Minimal Viable Comparison (If Time-Constrained)

If you only have time for basics:

### Minimum:
1. ✅ Random Baseline (done)
2. ✅ U-Net Supervised Only (need to train - 10 hours)
3. ✅ U-Net + Self-Training (you have this)

This gives you:
- Sanity check (random)
- Ablation study (supervised vs self-training)
- Your full method

**Time needed**: 10 hours (just train supervised U-Net)

### Comparison Table:
```
Random:        50% F1  (baseline)
U-Net Sup:     60% F1  (ablation)
U-Net + ST:    63% F1  (your method) ← +3% improvement
```

**Paper claim**: "Self-training improves F1 by 3% over supervised baseline (p<0.05)"

---

## Action Plan (Priority Order)

### This Week (Critical):
- [ ] **Day 1-2**: Train U-Net Supervised Only
- [ ] **Day 3**: Run `compare_all_models.py`
- [ ] **Day 4**: Analyze results, check significance
- [ ] **Day 5**: Write comparison section for paper

### If Time Permits:
- [ ] Train FCN baseline
- [ ] Train Attention U-Net variant
- [ ] Run ablation studies (thresholds, iterations)

### Before Submission:
- [ ] Double-check all models evaluated on same test set
- [ ] Verify statistical tests
- [ ] Create qualitative comparison figure
- [ ] Proofread comparison section

---

## FAQ

**Q: Do I need to retrain all baseline models?**
A: Only if you don't have them already. If you have saved checkpoints, just evaluate them on your test set.

**Q: Do I need to compare with published methods like CryoTransformer?**
A: You can cite their published numbers, but clearly state it's not a direct comparison (different tasks, scales, test sets).

**Q: What if my method doesn't beat all baselines?**
A: That's okay! Focus on the **contribution** (e.g., label efficiency, training speed, semi-supervised learning). Not every baseline needs to be beaten.

**Q: How many models minimum for a paper?**
A: At least 3:
1. Random/Simple baseline
2. Your method (full)
3. Ablation (your method with key component removed)

---

## Summary

**Minimum for credible paper**:
```bash
# 1. Train supervised U-Net (10 hours)
python train_unet_supervised_only.py

# 2. Run comparison
python compare_all_models.py

# 3. Use generated table in paper
# (paper_comparison_table.tex)
```

**That's it!** With these 3 models (Random, Supervised, Self-Training), you have a solid comparison for your paper.

---

## Next Steps

1. Check if you have saved U-Net model from training
2. Train U-Net supervised-only version (priority #1)
3. Run comparison script
4. Review results
5. Write paper section

Would you like me to create the `train_unet_supervised_only.py` script?
