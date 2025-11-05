# CryoEM Particle Picking with FixMatch + PU Learning

A modular PyTorch implementation of semi-supervised particle picking for Cryo-EM using FixMatch and Positive-Unlabeled (PU) Learning.

## Overview

This project implements a novel approach to Cryo-EM particle picking by combining:
- **FixMatch**: A semi-supervised learning technique using consistency regularization
- **PU Learning**: Learning from positive and unlabeled data with false positive penalties
- **Custom sampling**: Enforcing 1:4:1 ratio for Positive:Unlabeled:False Positive patches

## Features

- ✅ Modular PyTorch implementation
- ✅ Configurable via YAML files
- ✅ Custom data sampler for P:U:FP ratio enforcement
- ✅ Comprehensive evaluation metrics
- ✅ TensorBoard logging
- ✅ Model checkpointing and resume capabilities
- ✅ Inference on new micrographs
- ✅ Multiple output formats (CSV, STAR, BOX)

## Project Structure

```
CryoEM_FixMatch_PU/
├── configs/                  # Configuration files
│   ├── fixmatch_pu.yaml     # Main training config
│   └── preprocessing.yaml    # Preprocessing config
├── data/                     # Data directory
│   ├── raw/                  # Symlink to CryoPPP dataset
│   ├── processed/            # Preprocessed patches
│   └── splits/               # Train/val/test splits
├── datasets/                 # Dataset implementations
│   ├── cryoem_dataset.py     # Main dataset class
│   └── transforms.py         # Data augmentations
├── models/                   # Model architectures
│   ├── backbone/             # Backbone networks
│   │   └── resnet.py         # ResNet variants
│   ├── fixmatch_pu.py        # Main model
│   └── losses.py             # Loss functions
├── utils/                    # Utility modules
│   ├── logger.py             # Logging utilities
│   └── metrics.py            # Evaluation metrics
├── preprocessing.py          # Data preprocessing
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference script
└── experiments/              # Experiment outputs
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install numpy scipy pandas
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install pyyaml
pip install zarr numcodecs
pip install opencv-python
pip install tensorboard
```

### Optional Dependencies

```bash
# For advanced augmentations
pip install albumentations

# For EfficientNet backbone
pip install timm

# For Weights & Biases logging
pip install wandb
```

## Quick Start

### 1. Data Preparation

First, preprocess the CryoPPP dataset:

```bash
# Edit preprocessing config if needed
vim configs/preprocessing.yaml

# Run preprocessing
python preprocessing.py
```

This will:
- Extract patches from micrographs
- Create Positive, Unlabeled, and False Positive patches
- Save in Zarr format for efficient loading

### 2. Training

Train the FixMatch + PU model:

```bash
# Train with default config
python train.py --config configs/fixmatch_pu.yaml

# Train with custom settings
python train.py \
  --config configs/fixmatch_pu.yaml \
  --exp-name my_experiment \
  --device cuda:0 \
  --num-workers 8
```

### 3. Monitoring

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments/
```

### 4. Evaluation

Evaluate the trained model:

```bash
# Evaluate on test set
python evaluate.py \
  --checkpoint experiments/my_experiment/checkpoints/best_model.pth \
  --split test \
  --visualize
```

### 5. Inference

Run inference on new micrographs:

```bash
# Single micrograph
python inference.py \
  --checkpoint experiments/my_experiment/checkpoints/best_model.pth \
  --input path/to/micrograph.jpg \
  --output-dir results \
  --visualize

# Directory of micrographs
python inference.py \
  --checkpoint experiments/my_experiment/checkpoints/best_model.pth \
  --input path/to/micrographs/ \
  --output-dir results \
  --format star  # For RELION
```

## Configuration

### Main Training Config (`configs/fixmatch_pu.yaml`)

```yaml
# Data settings
data:
  patch_size: 128
  batch_ratio: [1, 4, 1]  # P:U:FP ratio

# Model architecture
model:
  backbone: resnet18
  use_ema: true
  ema_decay: 0.999

# Loss weights
loss:
  prior: 0.3  # Estimated positive ratio
  lambda_consistency: 1.0
  lambda_entropy: 0.1
  gamma_fp: 2.0  # FP penalty weight

# Training parameters
training:
  epochs: 100
  batch_size: 60
```

## Key Components

### 1. PU Loss (nnPU)

Implements non-negative PU learning loss:

```python
L_PU = π_p * E_P[ℓ(f(x), 1)] + max{E_U[ℓ(f(x), 0)] - π_p * E_P[ℓ(f(x), 0)], 0}
```

### 2. FixMatch Consistency

Enforces consistency between weak and strong augmentations:

```python
L_con = E_U[1(max q(x_w) ≥ τ) * CE(q(x_w), p(x_s))]
```

### 3. Custom Sampler

Ensures balanced batches with 1:4:1 ratio:

```python
# Each batch of 60 samples contains:
- 10 Positive patches
- 40 Unlabeled patches
- 10 False Positive patches
```

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Classification**: AUC, Precision, Recall, F1
- **Detection**: Coordinate-based matching metrics
- **PU-specific**: Contamination estimation, score distributions

## Tips for Best Results

### Data Quality
- Ensure particle coordinates are accurate
- Include diverse false positives (ice, aggregates, edges)
- Use sufficient unlabeled data for semi-supervised learning

### Hyperparameter Tuning
- Adjust `prior` based on estimated positive ratio in unlabeled data
- Tune `consistency_threshold` (0.90-0.95 typically works well)
- Scale `gamma_fp` based on false positive prevalence

### Training Strategy
- Use learning rate warmup for stability
- Monitor pseudo-label acceptance rate
- Check for mode collapse in unlabeled predictions

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Decrease `patch_size`
- Use gradient accumulation

### Poor Performance
- Check data quality and labels
- Increase `rampup_epochs` for consistency loss
- Adjust `prior` estimate
- Try different backbone (resnet34, resnet50)

### Slow Training
- Increase `num_workers` for data loading
- Use mixed precision training
- Enable `cudnn.benchmark`

## Citation

If you use this code, please cite:

```bibtex
@software{cryoem_fixmatch_pu,
  title = {CryoEM Particle Picking with FixMatch + PU Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cryoem-fixmatch-pu}
}
```

## References

- [FixMatch: Simplifying Semi-Supervised Learning](https://arxiv.org/abs/2001.07685)
- [Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://arxiv.org/abs/1703.00593)
- [CryoPPP Dataset](https://github.com/cryoppp/cryoppp_lite)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- CryoPPP team for the dataset
- PyTorch community for the framework
- Authors of FixMatch and PU Learning papers

---

For questions or issues, please open a GitHub issue or contact the maintainers.