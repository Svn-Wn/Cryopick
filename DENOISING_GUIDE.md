# Denoising Methods Guide for CryoEM Preprocessing

## Overview

The preprocessing pipeline supports multiple denoising methods to handle various types of noise in Cryo-EM micrographs. The denoising method can be configured in `configs/preprocessing.yaml`.

## Available Denoising Methods

### 1. **Gaussian** (Default)
```yaml
preprocessing:
  denoising_method: gaussian
```
- **Description**: Simple Gaussian blur filtering
- **Speed**: Very fast
- **Use case**: General-purpose noise reduction, good baseline
- **Parameters**: 5x5 kernel with automatic sigma

### 2. **NL-Means** (Non-Local Means)
```yaml
preprocessing:
  denoising_method: nl_means
```
- **Description**: Advanced denoising that preserves edges by comparing similar patches
- **Speed**: Slower than Gaussian
- **Use case**: High noise levels, when edge preservation is important
- **Parameters**: h=10, templateWindowSize=7, searchWindowSize=21

### 3. **Wiener**
```yaml
preprocessing:
  denoising_method: wiener
```
- **Description**: Frequency-domain filtering using FFT
- **Speed**: Moderate
- **Use case**: Periodic noise, known noise characteristics
- **Parameters**: 9x9 Gaussian kernel, K=0.01 noise ratio

### 4. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
```yaml
preprocessing:
  denoising_method: clahe
```
- **Description**: Enhances local contrast while limiting noise amplification
- **Speed**: Fast
- **Use case**: Low contrast images, uneven illumination
- **Parameters**: clipLimit=2.0, tileGridSize=(16, 16)

### 5. **Guided**
```yaml
preprocessing:
  denoising_method: guided
```
- **Description**: Edge-preserving smoothing using the image as its own guide
- **Speed**: Moderate
- **Use case**: Structure preservation with noise reduction
- **Parameters**: radius=20, epsilon=0.1

### 6. **Combined**
```yaml
preprocessing:
  denoising_method: combined
```
- **Description**: Full pipeline combining all methods sequentially
- **Speed**: Slowest
- **Use case**: Severely noisy images requiring aggressive denoising
- **Pipeline**: Gaussian → NL-Means → Wiener → CLAHE → Guided

## Configuration Example

Edit `configs/preprocessing.yaml`:

```yaml
# Image preprocessing
preprocessing:
  denoising_method: nl_means  # Choose from: gaussian, nl_means, wiener, clahe, guided, combined
  normalize: true
  standardize: true
  clip_std: 3

# Other settings remain the same
patches:
  patch_size: 128
  min_particle_distance: 50
  patches_per_image: 60
  batch_ratio: [1, 4, 1]
```

## Usage Examples

### Command Line

```bash
# Using default (gaussian)
python preprocessing.py

# The method will be read from configs/preprocessing.yaml
```

### Python Script

```python
from preprocessing import CryoEMPreprocessor
from pathlib import Path

# Configure with specific denoising method
config = {
    'patch_size': 128,
    'min_particle_distance': 50,
    'sigma': 5.0,
    'patches_per_image': 60,
    'batch_ratio': [1, 4, 1],
    'denoising_method': 'nl_means'  # Specify method here
}

preprocessor = CryoEMPreprocessor(config)
metadata = preprocessor.process_dataset(
    data_root=Path('/home/uuni/cryoppp/cryoppp_lite'),
    output_dir=Path('data/processed'),
    max_datasets=10
)

# Check which method was used
print(f"Denoising method used: {metadata['denoising_method']}")
```

## Testing Different Methods

Use the provided test script to compare all methods:

```bash
python test_denoising.py
```

This will:
1. Apply all denoising methods to a sample micrograph
2. Save a comparison image (`denoising_comparison.png`)
3. Test preprocessing with each method
4. Report statistics for each method

## Performance Comparison

| Method | Speed | Edge Preservation | Noise Reduction | Memory Usage |
|--------|-------|------------------|-----------------|--------------|
| Gaussian | ⚡⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Low |
| NL-Means | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| Wiener | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| CLAHE | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Low |
| Guided | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium |
| Combined | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High |

## Recommendations

### For Different Noise Types

- **Shot noise (random)**: Use `nl_means` or `wiener`
- **Gaussian noise**: Use `gaussian` or `nl_means`
- **Ice contamination**: Use `clahe` or `combined`
- **Low contrast**: Use `clahe` followed by other methods
- **Mixed noise**: Use `combined`

### For Different Use Cases

- **Quick preprocessing**: Use `gaussian`
- **High-quality training data**: Use `nl_means` or `combined`
- **Real-time inference**: Use `gaussian` or `clahe`
- **Research/experimentation**: Try all methods and compare

## Fallback Behavior

If a denoising method fails or is not recognized:
1. The system automatically falls back to Gaussian denoising
2. A warning message is printed
3. Processing continues without interruption

## Verification

The denoising method used is always recorded in:
1. `metadata.json` file in the output directory
2. The `.pt` file containing preprocessed data
3. Console output during preprocessing

You can verify which method was used:

```bash
# Check metadata
cat data/processed/metadata.json | grep denoising_method

# Or in Python
import json
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f"Denoising method: {metadata['denoising_method']}")
```

## Notes

- All methods preserve the [0, 1] normalization range
- Standardization is applied after denoising
- The choice of method does not affect P/U/FP patch extraction logic
- Denoising is applied to the entire micrograph before patch extraction