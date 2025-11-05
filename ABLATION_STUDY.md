# Ablation Study: PU vs PUFP Training

## Overview
This ablation study compares two training modes for CryoEM particle picking using Selective FixMatch + PU Learning:

1. **PU Mode**: Uses only Positive and Unlabeled samples
2. **PUFP Mode**: Uses Positive, Unlabeled, and False Positive samples

## Implementation Details

### Configurable Components

#### 1. Dataset (`datasets/cryoem_dataset.py`)
- Added `training_mode` parameter to control sample inclusion
- PU mode: Excludes false positive patches entirely
- PUFP mode: Includes all three sample types

```python
# PU mode - excludes false positives
if self.training_mode == 'PUFP':
    for patch in self.false_positive_patches:
        patches.append(patch)
        # FP samples included
```

#### 2. Batch Sampler (`PUFPBatchSampler`)
- Adjusts batch composition based on mode
- PU mode: Samples only from positive and unlabeled
- PUFP mode: Samples from all three types

```python
if training_mode == 'PU':
    ratio_sum = ratio[0] + ratio[1]  # P + U only
    self.num_fp = 0
else:  # PUFP
    ratio_sum = sum(ratio)  # P + U + FP
```

#### 3. Loss Function (`SelectiveFixMatchLoss`)
- Conditionally applies FP loss term
- PU mode: PU loss computed only on P/U samples
- PUFP mode: Full PU loss including FP penalty

```python
if self.training_mode == 'PU':
    pu_mask = (sample_types != 0)  # Exclude FP
    pu_loss = self.pu_loss(outputs[pu_mask], ...)
```

### Configuration (`configs/ablation_study.yaml`)

```yaml
data:
  training_mode: "PUFP"  # or "PU"
  batch_ratio_pu: [1, 4]      # For PU mode
  batch_ratio_pufp: [2, 3, 1] # For PUFP mode

loss:
  gamma_fp: 1.0  # FP weight (ignored in PU mode)
```

## Running the Ablation Study

### Option 1: Run Both Experiments Automatically
```bash
./run_ablation_study.sh
```

This will:
1. Train PU-only model
2. Train PUFP model
3. Generate comparison table

### Option 2: Run Individual Experiments

#### PU-only Training:
```bash
python train_ablation_study.py \
  --config configs/ablation_study.yaml \
  --mode PU \
  --exp-name ablation_PU_only \
  --num-workers 0
```

#### PUFP Training:
```bash
python train_ablation_study.py \
  --config configs/ablation_study.yaml \
  --mode PUFP \
  --exp-name ablation_PUFP \
  --num-workers 0
```

## Expected Results

Based on the CryoTransformer dataset characteristics:

### PU Mode Expectations:
- **Pros**: Cleaner learning signal, no mislabeled negatives
- **Cons**: Less diverse negative examples
- **Expected**: Higher precision, potentially lower recall

### PUFP Mode Expectations:
- **Pros**: More negative examples for better boundary learning
- **Cons**: Potential noise from mislabeled FP samples
- **Expected**: Better generalization, higher recall

## Output Structure

```
experiments/
├── ablation_PU_only/
│   ├── config.yaml
│   ├── best_model.pt
│   ├── results.json
│   └── tensorboard/
└── ablation_PUFP/
    ├── config.yaml
    ├── best_model.pt
    ├── results.json
    └── tensorboard/
```

## Results Analysis

Results are saved in JSON format with:
- Training mode
- Best validation AUC
- Test metrics (AUC, precision, recall, F1)
- Complete training history

### Comparison Metrics:
- **AUC**: Overall discrimination ability
- **Precision**: Fraction of correct positive predictions
- **Recall**: Fraction of particles detected
- **F1 Score**: Harmonic mean of precision/recall

## Key Files

1. **`train_ablation_study.py`**: Main training script with mode support
2. **`datasets/cryoem_dataset.py`**: Modified dataset with mode-aware loading
3. **`configs/ablation_study.yaml`**: Configuration for both modes
4. **`run_ablation_study.sh`**: Automated comparison script

## Hypothesis

We expect PUFP training to perform better overall because:
1. False positives provide hard negative examples
2. Helps the model learn better decision boundaries
3. The 70/30 split ensures sufficient unlabeled data for FixMatch

However, if the false positive labels are noisy, PU-only might perform better by avoiding misleading supervision.