# Active Learning Protocol: Attention U-Net for CryoEM Particle Picking

## 🎯 Objective

Demonstrate that **active learning** can achieve comparable or better performance than supervised learning while using **significantly fewer labeled samples** (~50-70% reduction).

---

## 🔬 Active Learning Framework

### What is Active Learning?

Active learning is a machine learning paradigm where the model **selectively queries** the most informative samples for labeling, rather than randomly selecting samples.

**Key Principle**: Not all unlabeled samples are equally valuable for training.

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  Initial Training Phase                     │
│  Train Attention U-Net on small labeled set (e.g., 10%)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Active Learning Loop (N iterations)            │
│                                                             │
│  1. Inference on Unlabeled Pool                            │
│     → Run model on all unlabeled samples                   │
│                                                             │
│  2. Acquisition Function                                    │
│     → Score samples by informativeness                      │
│     → Select top K most informative samples                 │
│                                                             │
│  3. Oracle Labeling (Simulated)                            │
│     → Retrieve ground truth labels for selected samples     │
│                                                             │
│  4. Update Training Set                                     │
│     → Add newly labeled samples to training set            │
│                                                             │
│  5. Retrain Model                                           │
│     → Train on expanded labeled set                         │
│                                                             │
│  6. Evaluate                                                │
│     → Measure performance on held-out test set             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Final Evaluation                          │
│  Compare with supervised baseline (all data labeled)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Experimental Design

### Dataset Split

Total: 5,706 images (5,172 train + 534 val)

**For Active Learning:**
- **Initial Labeled Set**: 517 images (10% of training data)
- **Unlabeled Pool**: 4,655 images (90% of training data)
- **Validation Set**: 534 images (held-out, never used for training)
- **Test Set**: Same as validation (for fair comparison)

### Active Learning Iterations

| Iteration | Labeled Samples | % of Total | Query Size |
|-----------|----------------|------------|------------|
| 0 (Initial) | 517 | 10% | - |
| 1 | 1,034 | 20% | 517 |
| 2 | 1,551 | 30% | 517 |
| 3 | 2,069 | 40% | 517 |
| 4 | 2,586 | 50% | 517 |
| 5 | 3,103 | 60% | 517 |
| 6 | 3,620 | 70% | 517 |
| 7 | 4,137 | 80% | 517 |
| 8 | 4,655 | 90% | 518 |
| 9 | 5,172 | 100% | 517 |

**Total iterations**: 10 (including initial)

---

## 🎲 Acquisition Functions

### 1. Uncertainty Sampling (Baseline)

**Least Confidence**: Select samples where model is least confident

```python
score = 1 - max(p_positive, p_negative)
```

**Rationale**: Uncertain samples are near decision boundary → most informative

### 2. Entropy-Based Sampling

**Max Entropy**: Select samples with highest prediction entropy

```python
score = -p * log(p) - (1-p) * log(1-p)
```

**Rationale**: High entropy = model is confused → informative

### 3. Margin Sampling

**Min Margin**: Select samples with smallest margin between classes

```python
score = 1 - |p_positive - p_negative|
```

**Rationale**: Small margin = decision is close → informative

### 4. Diversity Sampling (Advanced)

**Core-Set Selection**: Select diverse samples using k-means clustering in feature space

**Rationale**: Ensures coverage of input distribution → prevents redundancy

### 5. Hybrid: Uncertainty + Diversity

**Balanced Selection**:
- 70% selected by uncertainty
- 30% selected by diversity

**Rationale**: Combines informativeness with representativeness

---

## 📈 Comparison Baselines

### Baseline 1: Random Sampling
- Select samples uniformly at random
- Control for active learning effectiveness

### Baseline 2: Full Supervision
- Train on all 5,172 images from start
- Upper bound on performance

### Baseline 3: Progressive Supervised
- Train on increasing subsets (10%, 20%, ..., 100%)
- Same data quantities as active learning, but random selection

---

## 🔧 Training Configuration

### Fixed Hyperparameters (All Methods)

| Parameter | Value |
|-----------|-------|
| **Model** | Attention U-Net (31.4M params) |
| **Epochs per iteration** | 20 (initial), 10 (subsequent) |
| **Batch size** | 16 |
| **Learning rate** | 0.001 (initial), 0.0005 (fine-tuning) |
| **Optimizer** | Adam (β1=0.9, β2=0.999) |
| **Loss** | CombinedLoss (Focal + Dice) |
| **Random seed** | 42 |
| **Hardware** | 2× RTX A6000 with DataParallel |

### Why Different Epochs?

- **Initial training (20 epochs)**: Learn from scratch on small set
- **Subsequent iterations (10 epochs)**: Fine-tune on expanded set

---

## 📊 Evaluation Metrics

### Primary Metrics
1. **F1 Score** - Primary metric (balanced precision/recall)
2. **Area Under Learning Curve** - Total performance across iterations
3. **Sample Efficiency** - Performance at 50% labeled data

### Secondary Metrics
- Precision, Recall, IoU, AUC (standard)
- Annotation cost reduction (% fewer samples needed)

### Performance Curves

**Learning Curve**: F1 score vs. % labeled data
- Active learning should be above random sampling
- Should approach full supervision faster

---

## 🎯 Success Criteria

### Hypothesis

Active learning will achieve:
1. **Equal performance** to full supervision with **50% fewer samples**
2. **Better performance** than random sampling at all data percentages
3. **Faster convergence** (higher area under learning curve)

### Statistical Significance

- Run each method with 3 different random seeds
- Report mean ± std for all metrics
- Differences >3% F1 considered significant

---

## 📂 Output Structure

```
experiments/active_learning/
├── uncertainty_sampling/
│   ├── iteration_0/
│   │   ├── model.pt
│   │   ├── metrics.json
│   │   └── labeled_indices.json
│   ├── iteration_1/
│   ├── ...
│   └── iteration_9/
│
├── entropy_sampling/
│   └── [same structure]
│
├── margin_sampling/
│   └── [same structure]
│
├── diversity_sampling/
│   └── [same structure]
│
├── hybrid_sampling/
│   └── [same structure]
│
├── random_sampling/
│   └── [same structure]
│
├── full_supervision/
│   └── [baseline results]
│
└── comparison_results/
    ├── learning_curves.png
    ├── performance_table.json
    └── efficiency_analysis.json
```

---

## ⏱️ Timeline Estimate

### Training Time per Strategy

| Method | Iterations | Time per Iter | Total Time |
|--------|-----------|---------------|------------|
| Initial (20 epochs) | 1 | ~3 hours | 3 hours |
| Active learning (10 epochs) | 9 | ~1.5 hours | 13.5 hours |
| **Total per strategy** | 10 | - | **~16.5 hours** |

### Full Experiment

- **6 strategies** × 16.5 hours = **~99 hours (~4 days)**
- **With parallelization** (2 strategies at once): **~50 hours (~2 days)**

---

## 🔬 Reproducibility

### Fixed Elements
- [x] Random seed: 42 (NumPy, PyTorch, CUDA)
- [x] Initial labeled set: Same 10% for all methods
- [x] Validation set: Fixed 534 images
- [x] Model architecture: Attention U-Net
- [x] Hyperparameters: Documented above

### Variable Elements
- Acquisition function (the variable being tested)
- Labeled set composition (changes per iteration)

---

## 📝 For Your Paper

### Methods Section Template

> **Active Learning Framework**: We implemented an active learning approach to reduce annotation requirements for CryoEM particle picking. Starting with 10% labeled data (517 images), we iteratively selected the most informative samples using five acquisition functions: uncertainty sampling, entropy-based sampling, margin sampling, diversity sampling (core-set), and a hybrid approach. At each of 10 iterations, we selected 517 new samples for labeling and retrained the Attention U-Net model. We compared active learning against random sampling and full supervision baselines. All methods used identical hyperparameters (batch size=16, learning rate=0.001, 20 initial epochs, 10 fine-tuning epochs per iteration). Performance was evaluated on a fixed held-out validation set of 534 images.

### Results Table Template

**Table: Active Learning Performance at 50% Labeled Data**

| Method | F1 Score | Precision | Recall | % Reduction |
|--------|----------|-----------|--------|-------------|
| Full Supervision (100%) | XX.X% | XX.X% | XX.X% | 0% (baseline) |
| **Uncertainty Sampling (50%)** | **XX.X%** | XX.X% | XX.X% | **50%** |
| Entropy Sampling (50%) | XX.X% | XX.X% | XX.X% | 50% |
| Margin Sampling (50%) | XX.X% | XX.X% | XX.X% | 50% |
| Diversity Sampling (50%) | XX.X% | XX.X% | XX.X% | 50% |
| Hybrid Sampling (50%) | XX.X% | XX.X% | XX.X% | 50% |
| Random Sampling (50%) | XX.X% | XX.X% | XX.X% | 50% |

---

## 🎓 Key Contributions

This active learning framework demonstrates:

1. **Sample Efficiency**: Achieve comparable performance with 50% fewer labeled samples
2. **Acquisition Function Comparison**: Systematic evaluation of 5 strategies
3. **Practical Impact**: Reduce annotation time from weeks to days
4. **Reproducible Protocol**: Complete documentation for reproducibility

---

## 📧 Implementation Notes

### Oracle Simulation

Since we have ground truth for all images, we **simulate** the oracle:
- Pretend labels are unknown
- Active learning selects samples
- Retrieve true labels from dataset

**In practice**: Human expert would label selected samples

### Computational Efficiency

- Store model features to avoid recomputation
- Cache predictions on unlabeled pool
- Use GPU for batch inference

---

## ✅ Quality Checklist

Before running experiments:

- [ ] Initial labeled set is randomly selected
- [ ] Same initial set used for all methods
- [ ] Validation set is held-out (never queried)
- [ ] Random seeds are fixed
- [ ] All hyperparameters documented
- [ ] Acquisition functions implemented correctly
- [ ] Baseline methods included

---

**Next Steps**: Implement training script, acquisition functions, and comparison tools.
