# CryoTransformer Chunked Preprocessing System

## Overview
This system handles preprocessing of the massive CryoTransformer dataset (6,192 micrographs → ~19M patches) using a chunked approach to avoid memory issues.

## Key Features
- **Memory-efficient**: Saves patches in chunks of 500k samples (configurable)
- **HDD storage**: All data saved to `/mnt/hdd1/uuni/fixmatch/cryotransformer_data/chunks` to avoid filling root disk
- **Resumable**: Can resume preprocessing if interrupted
- **Configurable sampling**: Support for different negative sampling ratios (10%, 30%, 50%, 100%)
- **Metadata tracking**: Automatically generates `metadata.json` with dataset statistics

## Components

### 1. `preprocess_chunked.py`
Main preprocessing script that:
- Extracts patches using sliding window (128×128, stride 64)
- Saves patches in chunks to manage memory
- Supports resuming from interruptions
- Generates metadata with statistics

### 2. `datasets/cryoem_chunked_dataset.py`
Dataset loader that:
- Loads chunks on-demand or caches in memory
- Provides balanced batch sampling
- Compatible with FixMatch + PU Learning pipeline

### 3. `configs/preprocessing_config.yaml`
Configuration file with:
- Data paths (source and HDD output)
- Preprocessing parameters
- Sampling profiles

### 4. `train_with_chunked_data.py`
Example training script showing how to use the chunked dataset

## Quick Start

### 1. Test the system
```bash
python test_chunked_pipeline.py
```

### 2. Run preprocessing (default 30% negative sampling)
```bash
python preprocess_chunked.py
```

### 3. Or use the launcher for different configurations
```bash
# 10% negative sampling (fastest, ~4GB)
python run_chunked_preprocessing.py --ratio 0.1

# 30% negative sampling (default, ~6GB)
python run_chunked_preprocessing.py --ratio 0.3

# 50% negative sampling (balanced, ~8GB)
python run_chunked_preprocessing.py --ratio 0.5

# 100% negative sampling (full dataset, ~12GB)
python run_chunked_preprocessing.py --ratio 1.0
```

### 4. Resume if interrupted
```bash
python preprocess_chunked.py  # Automatically resumes
```

### 5. Start fresh (delete existing chunks)
```bash
python preprocess_chunked.py --no-resume
```

## Training with Chunked Data

After preprocessing, train your model:
```bash
python train_with_chunked_data.py
```

With custom settings:
```bash
python train_with_chunked_data.py \
    --batch-size 64 \
    --positive-ratio 0.2 \
    --epochs 100 \
    --load-to-memory  # If you have enough RAM
```

## Dataset Statistics

### Estimated sizes for different sampling ratios:
| Ratio | Total Patches | Disk Space | Chunks |
|-------|--------------|------------|--------|
| 10%   | ~7M          | ~4GB       | ~14    |
| 30%   | ~9.6M        | ~6GB       | ~20    |
| 50%   | ~12.3M       | ~8GB       | ~25    |
| 100%  | ~19M         | ~12GB      | ~38    |

## Memory Requirements

### During preprocessing:
- ~2-3GB RAM (processes images in batches, saves chunks incrementally)

### During training:
- **Without caching**: ~4GB RAM (loads chunks on-demand)
- **With caching** (`--load-to-memory`): ~8-16GB RAM depending on sampling ratio

## File Structure

```
/mnt/hdd1/uuni/fixmatch/cryotransformer_data/chunks/
├── metadata.json                 # Dataset statistics and chunk info
├── train_chunk_0000.pt           # Training chunks
├── train_chunk_0001.pt
├── ...
├── val_chunk_0000.pt             # Validation chunks
├── val_chunk_0001.pt
├── ...
└── test_chunk_0000.pt            # Test chunks
```

## Troubleshooting

### Out of memory during preprocessing
- Reduce `chunk_size` in config (default: 500000)
- Ensure `/mnt/hdd1` has enough space

### Training too slow
- Use `--load-to-memory` if you have RAM
- Increase `num_workers` for data loading
- Reduce `batch_size` if GPU memory limited

### Preprocessing interrupted
- Just run again - it will resume automatically
- Check `metadata.json` for progress

### Need to change sampling ratio
- Use `--no-resume` to start fresh with new ratio
- Or save to different output directory

## Integration with FixMatch + PU Learning

The chunked dataset is fully compatible with your existing training pipeline:
- Returns (weak_aug, strong_aug, label, sample_type) tuples
- Supports PU and PUFP training modes
- Works with ConsistencyLoss and PULoss
- Provides balanced batch sampling

## Performance Tips

1. **For fastest preprocessing**: Use ratio=0.1 (10% negatives)
2. **For best model performance**: Use ratio=0.3 or 0.5
3. **For full dataset**: Use ratio=1.0 (requires most resources)
4. **Resume capability**: Metadata saved after each chunk
5. **Parallel processing**: Use multiple workers in training

## Contact
If you encounter issues, check the logs or run the test suite first.