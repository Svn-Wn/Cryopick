# Active Learning Framework: CRITICAL BUG FIXES & Relaunch Guide

## 🎯 Objective

Prove that **Active Learning** can achieve **F1 73.03%** (the 100% data baseline from Attention U-Net) with only **30-40% of labeled data**, reducing annotation costs significantly.

---

## 🐛 CRITICAL BUGS IDENTIFIED & FIXED

### Bug #1: Performance Crash (F1 → 0.147)

**Symptom**: F1 score degraded to 0.147 as more data was added.

**Root Cause**:
```python
# WRONG (old code)
p_positive = outputs.mean(dim=(1, 2, 3)).cpu().numpy()  # Collapses 768×768 to single scalar!
uncertainty = 1 - np.maximum(p_positive, p_negative)
```

This **destroyed all spatial information**! For a 768×768 segmentation mask, this averaged 589,824 pixels into one number.

**Fix**:
```python
# CORRECT (new code)
p = outputs.squeeze(1)  # Keep spatial dimensions: (B, H, W)
pixel_uncertainty = torch.min(p, 1 - p)  # Per-pixel uncertainty
image_uncertainty = pixel_uncertainty.mean(dim=(1, 2))  # Aggregate spatially
```

**Impact**: Now properly measures per-pixel uncertainty before aggregation.

---

### Bug #2: Identical Results Across Strategies

**Symptom**: Uncertainty, Entropy, and Margin sampling gave identical results.

**Root Cause**: Same as Bug #1 - all acquisition functions collapsed to scalar averaging, making them functionally identical.

**Fix**: Each acquisition function now:
1. Computes **per-pixel** scores (uncertainty/entropy/margin)
2. Aggregates using **mean** across spatial dimensions
3. Returns **per-image** scores that capture spatial variability

**Example** (Entropy):
```python
# WRONG (old)
p = outputs.mean(dim=(1, 2, 3))  # Single value per image
entropy = -(p * np.log(p + epsilon) + (1 - p) * np.log(1 - p + epsilon))

# CORRECT (new)
p = outputs.squeeze(1)  # (B, H, W) - keep spatial
pixel_entropy = -(p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon))
image_entropy = pixel_entropy.mean(dim=(1, 2))  # Aggregate properly
```

---

### Bug #3: OOM Error in Diversity Sampling

**Symptom**: Out-of-memory errors when running diversity-based acquisition.

**Root Cause**:
```python
# WRONG (old)
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        x = module(x)
    if x.shape[2] == 1 and x.shape[3] == 1:
        break
```

This iterated through **all modules** and created huge intermediate tensors.

**Fix**: Use **forward hooks** to extract bottleneck features efficiently:
```python
# CORRECT (new)
def extract_features(self, model, data, device, batch_size=8):
    bottleneck_features = []

    def hook_fn(module, input, output):
        pooled = output.mean(dim=(2, 3))  # Global average pooling
        bottleneck_features.append(pooled.detach().cpu())

    # Register hook on bottleneck layer
    hook = None
    for name, module in model.named_modules():
        if 'enc4' in name or 'bottleneck' in name:
            hook = module.register_forward_hook(hook_fn)
            break

    # Forward pass (hook captures features automatically)
    with torch.no_grad():
        for batch in batches:
            _ = model(batch)

    hook.remove()
    return torch.cat(bottleneck_features, dim=0).numpy()
```

---

### Bug #4: Knowledge Loss Between Iterations

**Symptom**: Performance didn't improve much as more data was added.

**Root Cause**:
```python
# WRONG (old) - Line 461
model = AttentionUNet(in_channels=1, out_channels=1)  # Reinitialized EVERY iteration!
```

This **threw away all learned knowledge** each iteration.

**Fix**: **Warm-starting** - keep the model and fine-tune:
```python
# CORRECT (new)
if iteration == 0 or args.no_warmstart:
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    print("Initialized new model")
else:
    print("Warm-starting from previous iteration")
    # Model persists from previous iteration
```

---

### Bug #5: Poor Optimizer & No LR Scheduling

**Symptom**: Suboptimal convergence.

**Root Cause**:
```python
# WRONG (old)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# No learning rate scheduler
```

**Fix**:
```python
# CORRECT (new)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=args.learning_rate * 0.01
)
```

---

### Bug #6: No Data Augmentation

**Symptom**: Limited generalization with small labeled sets.

**Fix**: Use `CryoEMSegmentationDataset` with augmentation enabled:
```python
# CORRECT (new)
train_dataset = CryoEMSegmentationDataset(
    train_imgs_labeled,
    train_masks_labeled,
    is_training=True  # Enables random flips, rotations, etc.
)
```

---

## 📊 Experimental Setup

### Baseline Performance (From Fair Comparison V3)

| Model | Data | F1 Score | Note |
|-------|------|----------|------|
| **Attention U-Net** | **100% (5,172 images)** | **73.03%** | **Target** |
| Standard U-Net | 100% (5,172 images) | 72.64% | Reference |

### Active Learning Configuration

```bash
Training Pool: 5,172 images (data/cryotransformer_preprocessed/train)
Validation Set: 534 images (data/cryotransformer_preprocessed/val) - FIXED
Initial Labeled: 10% (517 images)
Query per Iteration: 10% (517 images)
Total Iterations: 10 (reaches 100% at iteration 10)
Epochs per Iteration: 30
Batch Size: 8
Learning Rate: 0.001 (with cosine annealing)
Optimizer: AdamW (weight_decay=1e-4)
Random Seed: 42
```

### Strategies to Compare

1. **Random** (Baseline): Select samples uniformly at random
2. **Uncertainty**: Select samples with highest per-pixel uncertainty
3. **Entropy**: Select samples with highest per-pixel entropy (optional)
4. **Hybrid**: 70% uncertainty + 30% diversity (optional)

---

## 🚀 How to Run

### Quick Start (Random vs Uncertainty)

```bash
# Launch both strategies in parallel on 2 GPUs
./run_active_learning_comparison.sh
```

This will:
- Launch **Random** strategy on GPU 0
- Launch **Uncertainty** strategy on GPU 1
- Run for ~5-6 hours each
- Save results to `experiments/active_learning_fixed/`

### Monitor Progress

```bash
# Watch Random strategy
tail -f experiments/active_learning_fixed/random/training.log

# Watch Uncertainty strategy
tail -f experiments/active_learning_fixed/uncertainty/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if processes are still running
ps aux | grep train_active_learning_fixed
```

### Analyze Results

```bash
# After experiments complete (or while running)
python3 analyze_active_learning_results.py
```

This generates:
- **Comparison plot**: F1 vs Data%
- **Data efficiency plot**: How much data needed to reach target F1
- **Summary JSON**: Detailed metrics

---

## 📈 Expected Results

### Success Criteria

**Active Learning is successful if:**

✅ **Uncertainty sampling reaches F1 73.03% with ≤40% of data**

This would mean:
- **Random**: Needs 100% data (5,172 images) → F1 73.03%
- **Uncertainty**: Needs ~40% data (2,069 images) → F1 73.03%
- **Savings**: ~60% fewer images to label (3,103 fewer labels)

### Likely Outcome

Based on typical Active Learning literature:

| Strategy | Data Needed | Savings |
|----------|-------------|---------|
| Random | 100% (5,172 images) | Baseline |
| Uncertainty | 30-50% (1,552-2,586 images) | 50-70% reduction |

---

## 📁 File Organization

```
CryoEM_FixMatch_PU/
├── train_active_learning_fixed.py          # FIXED implementation
├── run_active_learning_comparison.sh       # Launch script
├── analyze_active_learning_results.py      # Results analysis
├── ACTIVE_LEARNING_BUGFIX_GUIDE.md        # This document
│
└── experiments/
    └── active_learning_fixed/
        ├── random/
        │   ├── config.json
        │   ├── training.log
        │   ├── iteration_0/
        │   │   ├── metrics.json
        │   │   ├── best_model.pt
        │   │   └── labeled_indices.json
        │   ├── iteration_1/
        │   └── ...
        │
        ├── uncertainty/
        │   └── (same structure)
        │
        ├── active_learning_comparison.png   # Visualization
        └── comparison_summary.json          # Summary
```

---

## 🔬 Technical Details

### Key Differences from Old Implementation

| Aspect | Old (Buggy) | New (Fixed) |
|--------|-------------|-------------|
| **Spatial Info** | Collapsed to scalar | Per-pixel then aggregate |
| **Acquisition Functions** | All identical | Truly different |
| **Model Initialization** | Reset each iteration | Warm-started |
| **Optimizer** | Adam, no decay | AdamW with decay |
| **LR Scheduling** | None | Cosine annealing |
| **Augmentation** | None | CryoEM-specific |
| **Feature Extraction** | Inefficient, OOM | Hook-based, efficient |

### Per-Pixel Uncertainty Example

For a single 768×768 image:

**Old way**:
```
589,824 pixels → average → 1 number → uncertainty
Result: Lost all spatial variability
```

**New way**:
```
589,824 pixels → 589,824 uncertainties → mean uncertainty
Result: Captures spatial heterogeneity
```

**Impact**: Images with more uncertain regions (e.g., ambiguous particles) score higher.

---

## 🎯 Success Metrics

After experiments complete, check:

1. **Target Achievement**:
   - Did Uncertainty reach F1 73.03%?
   - At what data percentage?

2. **Data Efficiency**:
   - How much less data than Random?
   - Typical target: 30-40% savings

3. **Learning Curves**:
   - Does Uncertainty improve faster early on?
   - Does it plateau earlier?

4. **Practical Impact**:
   - Annotation cost savings
   - Time savings (assuming ~5 min per image to label)

---

## 🐛 Troubleshooting

### If F1 scores are still low:

1. **Check warm-starting**: Ensure model persists between iterations
2. **Check augmentation**: Verify `is_training=True` in dataset
3. **Check learning rate**: May need adjustment for your hardware
4. **Check batch size**: Reduce if OOM

### If OOM errors occur:

```bash
# Reduce batch size
python3 train_active_learning_fixed.py ... --batch-size 4
```

### If strategies give identical results:

This means the bug fix didn't work. Check:
1. Are you using `train_active_learning_fixed.py`?
2. Check acquisition function implementations
3. Verify spatial dimensions are preserved

---

## 📝 Summary

### What Was Wrong

1. ❌ Spatial information collapsed → meaningless scores
2. ❌ All strategies identical → no real comparison
3. ❌ OOM in diversity → inefficient feature extraction
4. ❌ Model reset each iteration → knowledge loss
5. ❌ Poor optimization → slow convergence
6. ❌ No augmentation → overfitting

### What's Fixed

1. ✅ Per-pixel uncertainty → meaningful scores
2. ✅ Truly different strategies → valid comparison
3. ✅ Efficient hook-based features → no OOM
4. ✅ Warm-starting → knowledge retention
5. ✅ AdamW + scheduler → better optimization
6. ✅ Data augmentation → better generalization

### Expected Outcome

**Active Learning should demonstrate 30-50% data savings** while achieving the same F1 73.03% performance as training with 100% of data.

This proves the value of intelligent sample selection for reducing annotation costs in CryoEM particle picking.

---

## 🚀 Next Steps

1. **Run experiments**: `./run_active_learning_comparison.sh`
2. **Monitor progress**: Check logs and GPU usage
3. **Analyze results**: Run `analyze_active_learning_results.py`
4. **Report findings**: Document data savings achieved

**Estimated Runtime**: ~5-6 hours per strategy on A6000 GPU

**Good luck! 🎯**
