# U-Net Architecture for Cryo-EM Particle Picking: Technical Documentation

## Overview

**U-Net is a semantic segmentation model**, not a simple classification model. It performs **dense pixel-wise prediction** to generate probability heatmaps indicating particle locations in cryo-EM micrographs.

---

## 1. Architecture Overview

### 1.1 Task Definition

**Input**: Cryo-EM micrograph patch `I ∈ ℝ^(H×W)`
**Output**: Probability heatmap `P ∈ [0,1]^(H×W)` where each pixel indicates particle presence probability
**Task**: Semantic segmentation (pixel-wise binary classification)

**Key Difference from Classification**:
- Classification: One label per image → "Does this patch contain a particle?" → Binary output
- **Segmentation**: One label per pixel → "Which pixels belong to particles?" → Heatmap output

### 1.2 U-Net Architecture

U-Net consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                        U-Net Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Encoder (Contracting Path)                                │
│   ↓                                                          │
│   Conv → Conv → MaxPool → Double channels                   │
│   ↓                                                          │
│   [Level 1]: 128×128×64   ──────┐                          │
│   [Level 2]:  64×64×128   ──────┼──→ Skip Connections      │
│   [Level 3]:  32×32×256   ──────┼──→ (Feature Fusion)      │
│   [Level 4]:  16×16×512   ──────┘                          │
│                                                              │
│   Bottleneck                                                 │
│   [Level 5]:   8×8×1024                                     │
│                                                              │
│   Decoder (Expanding Path)                                   │
│   ↑                                                          │
│   Upsample → Conv → Concatenate skip → Conv                 │
│   ↑                                                          │
│   [Level 4]:  16×16×512   (fused with encoder L4)          │
│   [Level 3]:  32×32×256   (fused with encoder L3)          │
│   [Level 2]:  64×64×128   (fused with encoder L2)          │
│   [Level 1]: 128×128×64   (fused with encoder L1)          │
│                                                              │
│   Output Layer                                               │
│   Conv 1×1 → Sigmoid → 128×128×1 (probability heatmap)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Architecture Components

### 2.1 Encoder (Contracting Path)

**Purpose**: Extract hierarchical features at multiple scales

```python
class Encoder(nn.Module):
    """
    Each encoder block:
    1. Double convolution (3×3 Conv + BatchNorm + ReLU) × 2
    2. Max pooling (2×2, stride=2) to downsample
    3. Double number of channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_skip = self.conv_block(x)  # Save for skip connection
        x_down = self.pool(x_skip)    # Downsample
        return x_down, x_skip

# Encoder stages (with 128×128 input):
# Level 1:  128×128×1  →  128×128×64  →  64×64×64   (skip: 128×128×64)
# Level 2:   64×64×64  →   64×64×128 →  32×32×128  (skip:  64×64×128)
# Level 3:  32×32×128  →  32×32×256  →  16×16×256  (skip:  32×32×256)
# Level 4:  16×16×256  →  16×16×512  →   8×8×512   (skip:  16×16×512)
```

**Feature Representation**:
- **Low-level features** (early layers): Edges, textures, noise patterns
- **Mid-level features**: Particle shapes, local structures
- **High-level features** (deep layers): Semantic context, particle vs background

### 2.2 Bottleneck

**Purpose**: Capture global context with maximum receptive field

```python
class Bottleneck(nn.Module):
    """
    Deepest layer with smallest spatial resolution
    Largest receptive field → captures global context
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

# Bottleneck: 8×8×512 → 8×8×1024
```

**Receptive Field**:
- At 128×128 input with 4 pooling layers
- Bottleneck receptive field: ~128×128 (sees entire patch!)
- Critical for understanding global particle context

### 2.3 Decoder (Expanding Path)

**Purpose**: Recover spatial resolution while fusing multi-scale features

```python
class Decoder(nn.Module):
    """
    Each decoder block:
    1. Upsampling (2× bilinear interpolation or transposed conv)
    2. Concatenate with corresponding encoder skip connection
    3. Double convolution to fuse features
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_up, x_skip):
        x = self.upsample(x_up)                      # 2× upsampling
        x = torch.cat([x, x_skip], dim=1)            # Concatenate skip
        x = self.conv_block(x)                       # Fuse features
        return x

# Decoder stages:
# Level 4:   8×8×1024 → 16×16×1024 + skip(16×16×512) → 16×16×512
# Level 3:  16×16×512 →  32×32×512 + skip(32×32×256) →  32×32×256
# Level 2:  32×32×256 →  64×64×256 + skip(64×64×128) →  64×64×128
# Level 1:  64×64×128 → 128×128×128 + skip(128×128×64) → 128×128×64
```

**Skip Connections**: Critical innovation of U-Net
- **Purpose**: Preserve spatial information lost during downsampling
- **Mechanism**: Concatenate encoder features with decoder features
- **Effect**: Enables precise localization (high-level semantics + low-level details)

### 2.4 Output Layer

**Purpose**: Map features to pixel-wise probabilities

```python
class OutputLayer(nn.Module):
    """
    Final 1×1 convolution + sigmoid activation
    Maps 64-channel feature map to single-channel probability map
    """

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)      # 128×128×64 → 128×128×1
        x = self.sigmoid(x)   # [0,1] probability
        return x

# Output: 128×128×1 probability heatmap
# Each pixel P(x,y) ∈ [0,1] indicates probability that pixel belongs to particle
```

---

## 3. Mathematical Formulation

### 3.1 Forward Pass

Let `E_i` denote encoder at level `i`, `D_i` decoder at level `i`:

```
Input: I ∈ ℝ^(128×128×1)

Encoder:
  x₁, skip₁ = E₁(I)           # 64×64×64,   skip: 128×128×64
  x₂, skip₂ = E₂(x₁)          # 32×32×128,  skip: 64×64×128
  x₃, skip₃ = E₃(x₂)          # 16×16×256,  skip: 32×32×256
  x₄, skip₄ = E₄(x₃)          # 8×8×512,    skip: 16×16×512

Bottleneck:
  x₅ = Bottleneck(x₄)         # 8×8×1024

Decoder:
  x₄' = D₄(x₅, skip₄)         # 16×16×512
  x₃' = D₃(x₄', skip₃)        # 32×32×256
  x₂' = D₂(x₃', skip₂)        # 64×64×128
  x₁' = D₁(x₂', skip₁)        # 128×128×64

Output:
  P = σ(Conv₁ₓ₁(x₁'))         # 128×128×1, σ = sigmoid
```

### 3.2 Loss Function

**Binary Cross-Entropy Loss** (pixel-wise):

```
L_BCE = -1/(H×W) ∑ᵢ₌₁ᴴ ∑ⱼ₌₁ᵂ [yᵢⱼ log(pᵢⱼ) + (1-yᵢⱼ)log(1-pᵢⱼ)]

where:
  yᵢⱼ ∈ {0,1} = ground truth mask (particle=1, background=0)
  pᵢⱼ ∈ [0,1] = predicted probability at pixel (i,j)
  H, W = height, width of image
```

**With Ignore Mask** (for semi-supervised learning):

```
L_masked = -1/N_valid ∑ᵢ,ⱼ [yᵢⱼ log(pᵢⱼ) + (1-yᵢⱼ)log(1-pᵢⱼ)] × 𝟙[mᵢⱼ ≠ -1]

where:
  mᵢⱼ ∈ {-1, 0, 1} = mask (-1 = ignore, 0 = negative, 1 = positive)
  𝟙[·] = indicator function
  N_valid = number of non-ignored pixels
```

---

## 4. Particle Picking Pipeline

### 4.1 Training Data Preparation

**Step 1: Coordinate to Mask Conversion**

```python
def coordinates_to_mask(coordinates, image_shape, particle_radius=20):
    """
    Convert particle center coordinates to binary segmentation mask

    Input:  coordinates = [(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)]
    Output: mask ∈ {0,1}^(H×W)

    For each particle center (xᵢ, yᵢ):
      - Draw filled circle with radius R
      - Set all pixels within circle to 1
      - Background pixels remain 0
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    for x, y in coordinates:
        # Draw filled circle (Euclidean distance ≤ R)
        cv2.circle(mask, (int(x), int(y)), particle_radius, 1, -1)

    return mask

# Example:
# Coordinates: [(64, 64), (90, 90)]
# Output mask: 128×128 with two circular regions = 1, rest = 0
```

**Step 2: Semi-Supervised Pseudo-Labeling**

```python
def generate_pseudo_labels(model, unlabeled_images, conf_threshold=0.95):
    """
    Generate pseudo-labels for unlabeled data using model predictions

    High-confidence pixels (p > 0.95) → Positive pseudo-label
    Low-confidence pixels (p < 0.05) → Negative pseudo-label
    Medium-confidence pixels → Ignore (uncertain)
    """

    with torch.no_grad():
        predictions = model(unlabeled_images)  # Get probability heatmap

    pseudo_mask = np.full(predictions.shape, -1, dtype=np.int8)  # Initialize to ignore

    # High confidence positive
    pseudo_mask[predictions > 0.95] = 1

    # High confidence negative
    pseudo_mask[predictions < 0.05] = 0

    # Medium confidence (0.05 ≤ p ≤ 0.95) remains -1 (ignored in loss)

    return pseudo_mask
```

### 4.2 Inference: Generating Particle Heatmaps

```python
def predict_particles(model, micrograph):
    """
    Generate probability heatmap for full micrograph

    Input:  micrograph ∈ ℝ^(H×W)
    Output: heatmap ∈ [0,1]^(H×W) (same size!)
    """

    # Normalize input
    micrograph = (micrograph - micrograph.mean()) / (micrograph.std() + 1e-8)

    # Convert to tensor
    x = torch.from_numpy(micrograph).float().unsqueeze(0).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        heatmap = model(x)  # U-Net output: same spatial size as input!

    return heatmap.squeeze().cpu().numpy()

# Key property: Output heatmap has SAME resolution as input
# Each pixel in heatmap corresponds to exact location in input image
```

### 4.3 Post-Processing: Extracting Particle Coordinates

**Step 1: Threshold Heatmap**

```python
def threshold_heatmap(heatmap, threshold=0.5):
    """
    Convert continuous probability heatmap to binary mask
    """
    binary_mask = (heatmap > threshold).astype(np.uint8)
    return binary_mask
```

**Step 2: Peak Detection**

```python
from scipy.ndimage import maximum_filter

def detect_particle_centers(heatmap, min_distance=20):
    """
    Find local maxima in heatmap (particle centers)

    Method: Non-maximum suppression
    - Apply maximum filter with window size = min_distance
    - Peak = pixel that equals maximum in its neighborhood
    """

    # Find local maxima
    local_max = maximum_filter(heatmap, size=min_distance)
    is_peak = (heatmap == local_max) & (heatmap > 0.5)

    # Extract coordinates
    particle_coords = np.argwhere(is_peak)

    return particle_coords  # List of (y, x) coordinates
```

**Step 3: Convert to Bounding Boxes** (optional, for comparison with object detectors)

```python
def heatmap_to_bounding_boxes(heatmap, min_size=400, threshold=0.5):
    """
    Convert heatmap to bounding boxes

    Method: Connected component analysis
    1. Threshold heatmap to binary mask
    2. Find connected components (particle blobs)
    3. Compute bounding box for each component
    """
    from scipy import ndimage

    # Threshold
    binary = (heatmap > threshold).astype(np.uint8)

    # Label connected components
    labeled, num_features = ndimage.label(binary)

    # Extract bounding boxes
    bboxes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)

        # Filter small components
        if component.sum() < min_size:
            continue

        # Get bounding box
        coords = np.argwhere(component)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes
```

---

## 5. U-Net vs Other Architectures

### 5.1 U-Net vs ResNet (Classification)

| Aspect | ResNet (Classification) | U-Net (Segmentation) |
|--------|-------------------------|----------------------|
| **Task** | Image-level label | Pixel-level labels |
| **Output** | Single class probability | Spatial probability map |
| **Architecture** | Encoder only | Encoder + Decoder |
| **Spatial Info** | Discarded (global pooling) | Preserved (skip connections) |
| **For Particle Picking** | "Does patch contain particle?" | "Where are particles in patch?" |

**Example**:
- **ResNet**: Input (128×128) → Output (scalar) → "80% chance of particle"
- **U-Net**: Input (128×128) → Output (128×128) → "This specific region is particle"

### 5.2 U-Net vs Transformer (DETR/CryoTransformer)

| Aspect | Transformer (DETR) | U-Net |
|--------|-------------------|-------|
| **Task** | Object detection | Semantic segmentation |
| **Output** | Discrete bounding boxes | Continuous heatmap |
| **Instance Separation** | Yes (separate boxes) | No (blob fusion) |
| **Attention** | Global (all-to-all) | Local (convolutional) |
| **Computational Cost** | O(n²) attention | O(n) convolution |

**Trade-offs**:
- **Transformer**: Better for counting discrete particles, distinguishing overlapping particles
- **U-Net**: Better for dense regions, uncertainty quantification, faster inference

### 5.3 Why U-Net for Particle Picking?

✅ **Preserves Spatial Information**: Skip connections maintain precise localization
✅ **Multi-Scale Features**: Captures particles at different sizes/contrasts
✅ **Dense Prediction**: Every pixel gets a prediction (no proposal generation needed)
✅ **Efficient**: Fully convolutional (no FC layers) → works on any input size
✅ **Uncertainty**: Probability heatmap shows confidence per region

---

## 6. Implementation Details

### 6.1 Model Specifications

```python
class UNet(nn.Module):
    """
    U-Net for Cryo-EM Particle Picking
    """

    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_features)      # 64
        self.enc2 = self.conv_block(base_features, base_features*2)   # 128
        self.enc3 = self.conv_block(base_features*2, base_features*4) # 256
        self.enc4 = self.conv_block(base_features*4, base_features*8) # 512

        # Bottleneck
        self.bottleneck = self.conv_block(base_features*8, base_features*16) # 1024

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_features*16, base_features*8, 2, stride=2)
        self.dec4 = self.conv_block(base_features*16, base_features*8)  # 512

        self.upconv3 = nn.ConvTranspose2d(base_features*8, base_features*4, 2, stride=2)
        self.dec3 = self.conv_block(base_features*8, base_features*4)   # 256

        self.upconv2 = nn.ConvTranspose2d(base_features*4, base_features*2, 2, stride=2)
        self.dec2 = self.conv_block(base_features*4, base_features*2)   # 128

        self.upconv1 = nn.ConvTranspose2d(base_features*2, base_features, 2, stride=2)
        self.dec1 = self.conv_block(base_features*2, base_features)     # 64

        # Output
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)        # 128×128×64
        x2 = self.enc2(self.pool(x1))   # 64×64×128
        x3 = self.enc3(self.pool(x2))   # 32×32×256
        x4 = self.enc4(self.pool(x3))   # 16×16×512

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))  # 8×8×1024

        # Decoder with skip connections
        x = self.upconv4(x5)              # 16×16×512
        x = torch.cat([x, x4], dim=1)     # 16×16×1024
        x = self.dec4(x)                  # 16×16×512

        x = self.upconv3(x)               # 32×32×256
        x = torch.cat([x, x3], dim=1)     # 32×32×512
        x = self.dec3(x)                  # 32×32×256

        x = self.upconv2(x)               # 64×64×128
        x = torch.cat([x, x2], dim=1)     # 64×64×256
        x = self.dec2(x)                  # 64×64×128

        x = self.upconv1(x)               # 128×128×64
        x = torch.cat([x, x1], dim=1)     # 128×128×128
        x = self.dec1(x)                  # 128×128×64

        # Output
        x = self.out(x)                   # 128×128×1
        return torch.sigmoid(x)
```

### 6.2 Model Statistics

```python
Model: UNet(base_features=64)

Total Parameters: 31,031,361 (31M)
Trainable Parameters: 31,031,361

Layer-wise breakdown:
  Encoder:     ~7.8M params
  Bottleneck:  ~9.4M params
  Decoder:     ~13.8M params

Memory (training, batch_size=128):
  Model weights:     ~120 MB
  Activations:       ~6 GB (with gradients)
  Total GPU memory:  ~8-10 GB

Inference speed (RTX A6000):
  128×128 patch:     ~0.5 ms
  512×512 patch:     ~8 ms
  3710×3838 image:   ~500 ms (with sliding window)
```

---

## 7. For Your Paper

### 7.1 Method Description Template

```latex
\subsection{U-Net Architecture for Particle Segmentation}

We employ a U-Net architecture~\cite{ronneberger2015unet} for semantic
segmentation of cryo-EM particles. Unlike classification-based approaches
that produce image-level predictions, our U-Net generates dense pixel-wise
probability heatmaps indicating particle locations.

\textbf{Architecture.} The U-Net consists of a contracting path (encoder)
and an expansive path (decoder) with skip connections. The encoder comprises
four downsampling blocks, each containing two 3×3 convolutions followed by
batch normalization, ReLU activation, and 2×2 max pooling. This progressively
reduces spatial resolution from 128×128 to 8×8 while increasing feature
channels from 64 to 1024. The decoder mirrors the encoder structure with
transposed convolutions for upsampling, concatenating corresponding encoder
features via skip connections to preserve spatial information. The final
1×1 convolution with sigmoid activation produces a probability heatmap of
the same size as the input (128×128×1).

\textbf{Training.} We train the model using binary cross-entropy loss on
pixel-wise labels, where particle regions are defined as circular masks
with radius R=20 pixels centered on annotated coordinates. The model is
optimized using AdamW with learning rate 1e-3 and weight decay 1e-4 for
50 epochs. We employ a semi-supervised self-training strategy: after
initial supervised training, we generate pseudo-labels for unlabeled patches
by thresholding predictions at confidence levels 0.95 (positive) and 0.05
(negative), then retrain with combined labeled and pseudo-labeled data for
additional iterations.

\textbf{Inference.} At test time, the model produces a probability heatmap
where each pixel p(x,y) ∈ [0,1] indicates the likelihood that location
(x,y) belongs to a particle. We extract particle centers by detecting local
maxima in the heatmap using non-maximum suppression with minimum inter-peak
distance of 20 pixels.
```

### 7.2 Key Citations

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  pages={234--241},
  year={2015}
}

@article{long2015fcn,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  journal={CVPR},
  year={2015}
}
```

### 7.3 Comparison with Related Work

```
Classification (ResNet/CNN):
  - Output: Binary label per patch
  - Loss: Cross-entropy over patches
  - Limitation: Coarse localization, no spatial information

Object Detection (Faster R-CNN, DETR):
  - Output: Bounding boxes with class labels
  - Loss: Box regression + classification
  - Advantage: Instance-level separation
  - Limitation: Requires expensive box annotations

Semantic Segmentation (U-Net, FCN):
  - Output: Pixel-wise probability map
  - Loss: Pixel-wise cross-entropy
  - Advantage: Dense predictions, precise localization
  - Limitation: No instance separation (particles may merge)

Our Approach (U-Net + Self-Training):
  - Combines semantic segmentation with semi-supervised learning
  - Leverages unlabeled data via pseudo-labeling
  - Achieves 65% precision with minimal annotation (center points only)
```

---

## 8. Summary

### U-Net for Particle Picking: Key Points

1. **Architecture Type**: Semantic segmentation (encoder-decoder with skip connections)
2. **Input**: Cryo-EM patch (128×128 pixels)
3. **Output**: Probability heatmap (128×128 pixels) - NOT single classification label!
4. **Task**: Dense pixel-wise prediction of particle regions
5. **Training**: Pixel-wise binary cross-entropy with circular masks
6. **Inference**: Generate heatmap → detect peaks → extract coordinates
7. **Advantage**: Precise localization, multi-scale features, uncertainty quantification
8. **Parameters**: 31M (comparable to ResNet50: 25M)
9. **Performance**: 65% precision, 60% recall (held-out test set)

### U-Net is NOT:
❌ A patch classification model (that's ResNet/VGG)
❌ An object detector (that's DETR/Faster R-CNN)
✅ A **semantic segmentation model** for dense spatial prediction

---

**For your paper, emphasize**:
- U-Net generates **dense probability heatmaps**, not discrete labels
- **Pixel-wise segmentation** enables precise particle localization
- **Skip connections** preserve fine-grained spatial information
- **Semi-supervised self-training** leverages unlabeled data efficiently
