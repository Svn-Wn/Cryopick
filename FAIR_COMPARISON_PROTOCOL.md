# Fair Comparison Protocol: Attention U-Net vs Standard U-Net

## Objective
Rigorously compare Attention U-Net and Standard U-Net architectures for CryoEM particle picking under **identical training conditions**.

---

## Experimental Design

### Models Under Comparison

1. **Standard U-Net**
   - 4 encoder levels + bottleneck + 4 decoder levels
   - Direct skip connections (concatenation)
   - 31,042,369 parameters

2. **Attention U-Net**
   - Same architecture as Standard U-Net
   - **Added**: Attention gates at each skip connection level
   - 31,393,901 parameters (+351,532, +1.13%)

### Key Difference
Attention gates learn to suppress irrelevant features in skip connections, focusing on particle regions.

---

## Training Configuration (IDENTICAL for Both Models)

### Data
- **Training set**: 5,172 images (768×768×3)
  - Source: `data/cryotransformer_preprocessed/train/`
  - 52 batch files × 100 images per batch

- **Validation set**: 534 images (768×768×3)
  - Source: `data/cryotransformer_preprocessed/val/`
  - 6 batch files × 100 images per batch (use first 534)

### Hyperparameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Epochs** | 100 | Standard for supervised learning |
| **Batch size** | 16 | Fits in GPU memory (2×A6000 with DataParallel) |
| **Learning rate** | 0.001 | Adam optimizer default |
| **Optimizer** | Adam | Standard for medical imaging |
| **Loss function** | CombinedLoss (Focal + Dice) | Handles class imbalance |
| **Random seed** | 42 | Reproducibility |

### Hardware
- **GPUs**: 2× NVIDIA RTX A6000 (48GB each)
- **Mode**: DataParallel (splits batch across GPUs)
- **CUDA**: Deterministic mode enabled

### Training Strategy
- **Phase**: Supervised learning only (no self-training)
- **Validation**: After each epoch on held-out set
- **Checkpointing**: Save best model based on validation F1 score
- **Metrics**: Precision, Recall, F1, IoU, AUC

---

## Evaluation Metrics

### Primary Metric
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

### Secondary Metrics
- **Precision**: Positive predictive value (fewer false positives = higher)
- **Recall**: True positive rate (fewer false negatives = higher)
- **IoU**: Intersection over Union (segmentation quality)
- **AUC**: Area under ROC curve (discriminative power)

---

## Fairness Checklist

### ✅ Same Data
- [ ] Both models train on identical 5,172 training images
- [ ] Both models validate on identical 534 validation images
- [ ] Same data preprocessing and augmentation

### ✅ Same Training Procedure
- [ ] Same number of epochs (100)
- [ ] Same batch size (16)
- [ ] Same learning rate (0.001)
- [ ] Same optimizer (Adam with same parameters)
- [ ] Same loss function (CombinedLoss with same weights)

### ✅ Same Hardware
- [ ] Both use 2× A6000 GPUs with DataParallel
- [ ] Same CUDA settings (deterministic mode)
- [ ] Same PyTorch version

### ✅ Same Random Seed
- [ ] Both use seed=42 for NumPy, PyTorch, and CUDA
- [ ] Ensures reproducible weight initialization

### ✅ Documentation
- [ ] All settings documented in training logs
- [ ] Command-line arguments recorded
- [ ] Training curves saved for inspection
- [ ] Final metrics computed on same validation set

---

## Expected Outcomes

### Hypothesis
Attention U-Net will achieve **higher precision** (fewer false positives) but potentially **lower recall** (may miss some particles) compared to Standard U-Net.

### Success Criteria
The model with **higher F1 score** on the validation set is considered superior for this task.

### Statistical Significance
Since this is a deterministic experiment (fixed seed), differences >2% in F1 score are considered meaningful.

---

## Training Scripts

### Standard U-Net
```bash
python3 train_standard_unet_fair_comparison.py \
  --train-data-dir data/cryotransformer_preprocessed/train \
  --val-data-dir data/cryotransformer_preprocessed/val \
  --output-dir experiments/fair_comparison/standard_unet \
  --initial-epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --device cuda:0 \
  --multi-gpu \
  --seed 42
```

### Attention U-Net
```bash
python3 train_attention_unet.py \
  --train-data-dir data/cryotransformer_preprocessed/train \
  --val-data-dir data/cryotransformer_preprocessed/val \
  --output-dir experiments/fair_comparison/attention_unet \
  --initial-epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --device cuda:0 \
  --multi-gpu \
  --seed 42
```

---

## Timeline

- **Training time per model**: ~15 hours (100 epochs × 9 min/epoch)
- **Total time**: ~30 hours for both models
- **Recommendation**: Train sequentially or use separate GPUs

---

## Output Structure

```
experiments/fair_comparison/
├── standard_unet/
│   ├── iteration_0_supervised/
│   │   ├── best_model.pt          # Best checkpoint (by val F1)
│   │   ├── model.pt                # Final model
│   │   ├── metrics.json            # Final metrics
│   │   ├── training_curve.png      # Loss/metrics over epochs
│   │   └── visualization.png       # Sample predictions
│   └── training.log
│
├── attention_unet/
│   ├── iteration_0_supervised/
│   │   ├── best_model.pt
│   │   ├── model.pt
│   │   ├── metrics.json
│   │   ├── training_curve.png
│   │   └── visualization.png
│   └── training.log
│
└── comparison_results.json         # Final comparison table
```

---

## Analysis Plan

After both models complete training:

1. **Load validation metrics** from `metrics.json`
2. **Compare F1 scores** (primary metric)
3. **Analyze precision-recall trade-off**
4. **Visualize predictions** side-by-side
5. **Statistical summary** in comparison table
6. **Document findings** for publication

---

## Publication-Ready Comparison Table

| Model | Parameters | Precision | Recall | F1 Score | IoU | AUC |
|-------|-----------|-----------|--------|----------|-----|-----|
| Standard U-Net | 31.0M | TBD | TBD | TBD | TBD | TBD |
| Attention U-Net | 31.4M | TBD | TBD | TBD | TBD | TBD |
| **Δ (Attention - Standard)** | +1.13% | TBD | TBD | TBD | TBD | TBD |

---

## Reproducibility Statement

All experiments conducted with:
- **PyTorch version**: [auto-detected]
- **CUDA version**: 12.2
- **Random seed**: 42
- **Hardware**: 2× NVIDIA RTX A6000 (48GB)
- **Dataset**: CryoTransformer preprocessed data (5,706 images total)
- **Date**: October 28, 2025

Full code, configurations, and results available in this repository.

---

## Contact

For questions about this comparison protocol, refer to the training scripts in this directory or check the training logs.

**Important**: Do not modify training parameters after starting experiments to ensure fairness!
