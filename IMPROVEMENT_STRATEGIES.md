# Strategies to Improve U-Net Performance

## Current Performance Gap

**Current U-Net**: Precision 65.09%, Recall 60.40%, F1 62.66%
**CryoTransformer**: Precision 76.25%, Recall ~72%, F1 74.0%
**Gap to Close**: ~11% in both Precision and F1

---

## Top 10 Improvement Strategies (Prioritized)

### 🥇 1. **Add Attention Mechanisms** (Expected +3-5% F1)

**Why**: Attention helps the model focus on relevant particle regions, similar to how Transformers work.

**Implementation Options**:
- **Attention U-Net**: Add attention gates in skip connections
- **Self-Attention**: Add spatial attention layers
- **Channel Attention**: SE-Net style squeeze-and-excitation blocks

**Code Changes**:
```python
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# In U-Net decoder, replace skip connections with attention
att = AttentionBlock(F_g=decoder_channels, F_l=encoder_channels, F_int=encoder_channels//2)
x = att(decoder_features, encoder_features)
```

**Effort**: Medium | **Impact**: High

---

### 🥈 2. **Improve Pseudo-Labeling Strategy** (Expected +2-4% F1)

**Current Issue**: Fixed confidence thresholds (0.95 positive, 0.05 negative) may be too strict/loose.

**Improvements**:
- **Dynamic thresholds**: Start conservative, gradually relax
- **Curriculum learning**: Easy samples first, hard samples later
- **Uncertainty-based selection**: Use model confidence + entropy
- **Ensemble pseudo-labels**: Use multiple model checkpoints

**Implementation**:
```python
def dynamic_threshold_schedule(iteration, total_iterations):
    """Gradually relax confidence thresholds"""
    # Start strict, become more permissive
    pos_thresh = 0.98 - (iteration / total_iterations) * 0.08  # 0.98 → 0.90
    neg_thresh = 0.02 + (iteration / total_iterations) * 0.08  # 0.02 → 0.10
    return pos_thresh, neg_thresh

def uncertainty_based_selection(predictions, threshold=0.95):
    """Select pseudo-labels based on both confidence and uncertainty"""
    confidence = torch.max(predictions, dim=1)[0]
    entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)

    # Select samples with high confidence AND low uncertainty
    high_conf = confidence > threshold
    low_entropy = entropy < 0.1  # Low uncertainty

    reliable_mask = high_conf & low_entropy
    return reliable_mask
```

**Effort**: Low | **Impact**: Medium-High

---

### 🥉 3. **Better Data Augmentation** (Expected +2-3% F1)

**Current Issue**: Limited augmentation may cause overfitting.

**Add Strong Augmentations**:
- **Geometric**: Random rotation (±15°), flip, elastic deformation
- **Intensity**: Gaussian noise, contrast/brightness adjustment
- **Domain-specific**: Simulated ice contamination, defocus variation
- **Mixup/CutMix**: Blend training samples

**Implementation**:
```python
import albumentations as A

augmentation = A.Compose([
    # Geometric
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),

    # Intensity
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),

    # Advanced
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
])
```

**Effort**: Low | **Impact**: Medium

---

### 4. **Use Better Loss Functions** (Expected +1-3% F1)

**Current**: Binary Cross-Entropy (BCE)

**Better Options**:
- **Focal Loss**: Focus on hard examples
- **Dice Loss**: Better for imbalanced segmentation
- **Combo Loss**: BCE + Dice + Focal
- **Class-balanced loss**: Weight by inverse class frequency

**Implementation**:
```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target, ignore_mask=None):
        if ignore_mask is not None:
            pred = pred[ignore_mask != -1]
            target = target[ignore_mask != -1]

        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return focal * (1 - self.dice_weight) + dice * self.dice_weight

# Use in training
criterion = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.3)
```

**Effort**: Low | **Impact**: Medium

---

### 5. **Increase Model Capacity** (Expected +2-4% F1)

**Current**: U-Net with base_features=64 (~31M params)

**Improvements**:
- Increase base features: 64 → 128
- Deeper encoder: 4 levels → 5 levels
- Add residual connections
- Use better encoder backbone (ResNet, EfficientNet)

**Implementation**:
```python
# Option A: Increase capacity
model = UNet(in_channels=1, out_channels=1, base_features=128)  # ~123M params

# Option B: Residual U-Net
class ResUNet(nn.Module):
    """U-Net with residual blocks"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Use ResNet as encoder
        from torchvision.models import resnet34
        self.encoder = resnet34(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)

        # Build decoder with skip connections
        # ... (implementation similar to U-Net)

# Option C: EfficientNet backbone
from segmentation_models_pytorch import Unet
model = Unet(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)
```

**Effort**: Medium | **Impact**: High
**Note**: Requires more GPU memory and training time

---

### 6. **Better Training Strategy** (Expected +1-2% F1)

**Improvements**:
- **Longer training**: 50+60 epochs → 100+100 epochs
- **Cosine annealing**: Better learning rate schedule
- **Warmup**: Gradual learning rate increase
- **Label smoothing**: Prevent overconfidence
- **Early stopping**: Based on validation F1, not loss

**Implementation**:
```python
# Cosine annealing with warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Warmup
def warmup_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

# Label smoothing
def label_smoothing(target, smoothing=0.1):
    target = target * (1 - smoothing) + 0.5 * smoothing
    return target
```

**Effort**: Low | **Impact**: Medium

---

### 7. **Use More Diverse Training Data** (Expected +3-5% F1)

**Current**: Single dataset subset (700k patches)

**Improvements**:
- **Full CryoPPP dataset**: Use all available data
- **Cross-protein training**: Include multiple protein types
- **Synthetic data**: Generate augmented particles
- **Hard negative mining**: Focus on difficult negatives

**Implementation**:
```python
# Load multiple protein datasets
datasets = [
    load_dataset('cryoppp_subset'),
    load_dataset('empiar_10081'),
    load_dataset('empiar_10093'),
]
combined_dataset = ConcatDataset(datasets)

# Hard negative mining
def mine_hard_negatives(model, unlabeled_data, top_k=0.1):
    """Select hardest negative samples (high false positive rate)"""
    model.eval()
    with torch.no_grad():
        predictions = model(unlabeled_data)
        # Select samples with high false positive probability
        false_pos_scores = predictions[:, 1]  # Positive class prob

        # Take top-k most confusing negatives
        num_hard = int(len(unlabeled_data) * top_k)
        hard_neg_indices = torch.argsort(false_pos_scores, descending=True)[:num_hard]

    return unlabeled_data[hard_neg_indices]
```

**Effort**: Medium | **Impact**: High
**Note**: Requires accessing more data or generating synthetic samples

---

### 8. **Ensemble Multiple Models** (Expected +2-4% F1)

**Strategy**: Train multiple models and combine predictions

**Approaches**:
- **Different initializations**: Train 3-5 models with different seeds
- **Different architectures**: U-Net + Attention U-Net + ResUNet
- **Different augmentations**: Each model sees different augmented data
- **Snapshot ensembles**: Save checkpoints during training

**Implementation**:
```python
# Train multiple models
models = [
    UNet(base_features=64),
    UNet(base_features=96),
    AttentionUNet(base_features=64),
]

# Ensemble prediction
def ensemble_predict(models, image):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(image))
            predictions.append(pred)

    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

# Or weighted ensemble
weights = [0.4, 0.3, 0.3]  # Based on validation performance
ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
```

**Effort**: High | **Impact**: Medium-High
**Note**: Requires training multiple models (more compute)

---

### 9. **Improve Negative Sampling** (Expected +1-3% F1)

**Current Issue**: Random unlabeled patches may not be informative

**Better Strategies**:
- **Balanced sampling**: Equal positive/negative per batch
- **Hard negative mining**: Focus on false positives
- **Informative region sampling**: Sample from particle-like regions
- **Boundary sampling**: Sample near particle edges

**Implementation**:
```python
class BalancedSampler(Sampler):
    def __init__(self, dataset, positive_ratio=0.5):
        self.dataset = dataset
        self.positive_ratio = positive_ratio

        # Separate positive and negative indices
        self.pos_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
        self.neg_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]

    def __iter__(self):
        # Sample equal numbers of positives and negatives
        num_samples = len(self.dataset)
        num_pos = int(num_samples * self.positive_ratio)
        num_neg = num_samples - num_pos

        pos_samples = random.choices(self.pos_indices, k=num_pos)
        neg_samples = random.choices(self.neg_indices, k=num_neg)

        indices = pos_samples + neg_samples
        random.shuffle(indices)
        return iter(indices)

# Use in DataLoader
sampler = BalancedSampler(train_dataset, positive_ratio=0.5)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
```

**Effort**: Low | **Impact**: Medium

---

### 10. **Post-Processing Refinement** (Expected +1-2% F1)

**Improvements**:
- **Morphological operations**: Remove small false positives
- **Connected component analysis**: Filter by size/shape
- **Non-maximum suppression**: Remove duplicate detections
- **Confidence thresholding**: Optimize threshold on validation set

**Implementation**:
```python
import cv2
from scipy import ndimage

def post_process_predictions(prob_map, min_size=50, threshold=0.5):
    """Refine predictions with morphological operations"""
    # Threshold
    binary_mask = (prob_map > threshold).astype(np.uint8)

    # Morphological opening (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Connected component analysis
    labeled, num_features = ndimage.label(binary_mask)

    # Filter small components
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if component.sum() < min_size:
            binary_mask[component] = 0

    return binary_mask

# Optimize threshold on validation set
def find_optimal_threshold(model, val_loader):
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        f1 = evaluate_with_threshold(model, val_loader, thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1
```

**Effort**: Low | **Impact**: Low-Medium

---

## Implementation Priority

### Quick Wins (1-2 days):
1. ✅ Better loss functions (Focal + Dice)
2. ✅ Improved data augmentation
3. ✅ Better training strategy (cosine annealing, warmup)
4. ✅ Post-processing refinement

**Expected Combined Gain**: +4-8% F1

### Medium Effort (3-7 days):
5. ✅ Add attention mechanisms
6. ✅ Improve pseudo-labeling strategy
7. ✅ Better negative sampling

**Expected Combined Gain**: +5-10% F1

### High Effort (1-2 weeks):
8. ✅ Increase model capacity (ResNet/EfficientNet backbone)
9. ✅ Use more diverse training data
10. ✅ Ensemble multiple models

**Expected Combined Gain**: +7-13% F1

---

## Recommended Action Plan

### Phase 1: Quick Improvements (Week 1)
```bash
# 1. Implement combined loss (Focal + Dice)
# 2. Add strong data augmentation
# 3. Use cosine annealing + warmup
# 4. Optimize prediction threshold

Expected improvement: +5-8% F1 → Target: 67-70% F1
```

### Phase 2: Architecture Upgrades (Week 2)
```bash
# 5. Add attention gates to U-Net
# 6. Implement dynamic pseudo-labeling thresholds
# 7. Use balanced sampling

Expected improvement: +3-5% F1 → Target: 70-73% F1
```

### Phase 3: Advanced Methods (Week 3-4)
```bash
# 8. Train with ResNet/EfficientNet backbone
# 9. Expand training data (full CryoPPP)
# 10. Train ensemble of 3-5 models

Expected improvement: +2-4% F1 → Target: 72-75% F1
```

**Total Expected Improvement**: +10-17% F1
**Potential Final Performance**: **72-75% F1** (matching CryoTransformer!)

---

## Critical Success Factors

1. ✅ **Validation strategy**: Always evaluate on held-out validation set
2. ✅ **Avoid overfitting**: Monitor train vs validation performance gap
3. ✅ **Incremental changes**: Implement one improvement at a time
4. ✅ **Ablation studies**: Measure impact of each change
5. ✅ **Resource budget**: Balance performance vs training time/cost

---

## Expected Final Results

If you implement all Quick Wins + Medium Effort improvements:

**Projected Performance**:
- Precision: 65% → **70-73%**
- Recall: 60% → **65-68%**
- F1-Score: 63% → **68-71%**

**Gap to CryoTransformer**: 11% → **3-6%**

This would make U-Net a **strong alternative** to CryoTransformer with better training efficiency!

---

## Next Steps

1. Start with Quick Wins (Focal loss + augmentation)
2. Monitor validation F1 after each change
3. Track improvements in a spreadsheet
4. Share results and iterate

Would you like me to implement any of these improvements?
