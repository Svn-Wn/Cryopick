# Quick Start: Run Fixed SSL with CryoTransformer Data

## ✅ Data Confirmed and Converted

**Source:** `/home/uuni/cryoppp/CryoTransformer/train_val_test_data/`
- **Train:** 5,172 images (using 1,000 for experiments)
- **Val:** 534 images
- **Test:** 486 images
- **Format:** JPEG images + CSV coordinates

**Converted to:**
- **Images:** `data/cryotransformer/images/` (1,000 images)
- **Coordinates:** `data/cryotransformer/coordinates.json`

---

## Quick Test (2-3 hours)

Test on 5% and 10% labels to verify everything works:

```bash
python3 train_ssl_evaluation_fixed.py \
  --image-dir data/cryotransformer/images \
  --coords-file data/cryotransformer/coordinates.json \
  --output-dir experiments/ssl_cryotransformer_quick \
  --labeled-ratios 0.05,0.10 \
  --max-images 1000 \
  --particle-radius 10
```

**Note:** `--particle-radius 10` is based on typical particle size. Adjust if needed.

---

## Full Evaluation (1-2 days)

Full evaluation on all label ratios:

```bash
python3 train_ssl_evaluation_fixed.py \
  --image-dir data/cryotransformer/images \
  --coords-file data/cryotransformer/coordinates.json \
  --output-dir experiments/ssl_cryotransformer_full \
  --labeled-ratios 0.05,0.10,0.20,0.50,1.0 \
  --max-images 1000 \
  --particle-radius 10
```

---

## Convert More Data (Optional)

### Convert all training data (5,172 images):
```bash
python3 convert_cryotransformer_data.py \
  --split train \
  --output-dir data/cryotransformer_full
```

### Convert all splits (train + val + test = 6,192 images):
```bash
python3 convert_cryotransformer_data.py \
  --split all \
  --output-dir data/cryotransformer_all
```

---

## Expected Timeline

### Quick Test (100 images, 2 ratios)
- 5% labels (5 labeled + 85 unlabeled):
  - Supervised training: 30 min
  - FixMatch training: 1-2 hours
- 10% labels (10 labeled + 80 unlabeled):
  - Likely skipped (supervised F1 > 95%)
- **Total: ~2-3 hours**

### Full Test (1000 images, 5 ratios)
- 5% labels: ~6-8 hours
- 10% labels: ~30 min (likely skipped)
- 20% labels: ~30 min (likely skipped)
- 50% labels: ~30 min (likely skipped)
- 100% labels: ~6-8 hours
- **Total: ~15-20 hours (1 day)**

---

## Monitoring Progress

### Check training status:
```bash
# Watch output directory
watch -n 10 'ls -lh experiments/ssl_cryotransformer_quick/'

# Check GPU usage
watch -n 2 nvidia-smi
```

### View results as they come:
```bash
# View latest metrics
cat experiments/ssl_cryotransformer_quick/ssl_eval_fixmatch_ratio_5.json

# View summary (when complete)
cat experiments/ssl_cryotransformer_quick/ssl_evaluation_fixmatch_summary.json
```

---

## What to Expect

### At 5% labels (45 labeled, 855 unlabeled):
```
USING SSL: Supervised F1 < 95%
→ FixMatch training will run
→ Expected +2-3% F1 improvement
```

### At 10%+ labels:
```
SKIPPING SSL: Supervised F1 > 95%
→ FixMatch skipped (early stopping)
→ 0% gain (uses supervised model)
```

---

## Troubleshooting

### Issue: Out of memory
```bash
# Reduce batch size
python3 train_ssl_evaluation_fixed.py \
  --batch-size 2 \
  ... (other args)
```

### Issue: Too slow
```bash
# Reduce max images
python3 train_ssl_evaluation_fixed.py \
  --max-images 500 \
  ... (other args)
```

### Issue: Different particle size
Check your CSV files for the "Diameter" column:
```bash
head -2 /home/uuni/cryoppp/CryoTransformer/train_val_test_data/particle_coordinates/16nov14y_16nov10d_00006sq_00001hl_00008ed-s-DW.csv | tail -1
```

Typical diameter: 84 pixels → radius ≈ 42 pixels / 4 = ~10 pixels for mask

Adjust `--particle-radius` accordingly.

---

## After Experiments Complete

### 1. View Results
```bash
cat experiments/ssl_cryotransformer_full/ssl_evaluation_fixmatch_summary.json
```

### 2. Compare with Original
```bash
# Original (broken) implementation
python3 train_ssl_evaluation.py \
  --image-dir data/cryotransformer/images \
  --coords-file data/cryotransformer/coordinates.json \
  --output-dir experiments/ssl_original

# Compare
python3 -c "
import json

with open('experiments/ssl_original/ssl_evaluation_summary.json') as f:
    original = json.load(f)

with open('experiments/ssl_cryotransformer_full/ssl_evaluation_fixmatch_summary.json') as f:
    fixed = json.load(f)

print('Comparison: Original vs Fixed')
print('='*60)
for o, f in zip(original, fixed):
    ratio = o['labeled_ratio']
    print(f'{ratio*100:.0f}% labels:')
    print(f'  Original SSL gain: {o[\"ssl_gain\"][\"f1_score\"]*100:+.2f}%')
    print(f'  Fixed SSL gain:    {f[\"ssl_gain\"][\"f1_score\"]*100:+.2f}%')
"
```

---

## Data Verification

```bash
# Check data format
python3 -c "
import json
import cv2

# Load coordinates
with open('data/cryotransformer/coordinates.json') as f:
    coords = json.load(f)

# Check first image
img_name = list(coords.keys())[0]
img = cv2.imread(f'data/cryotransformer/images/{img_name}', 0)

print(f'Image: {img_name}')
print(f'  Shape: {img.shape}')
print(f'  Type: {img.dtype}')
print(f'  Num particles: {len(coords[img_name])}')
print(f'  Sample coords: {coords[img_name][:3]}')
"
```

Expected output:
```
Image: 16nov14y_16nov10d_00006sq_00001hl_00008ed-s-DW.jpg
  Shape: (3838, 3710)
  Type: uint8
  Num particles: 92
  Sample coords: [[1970.0, 723.0], [2220.0, 1887.0], [665.0, 1964.0]]
```

---

## Summary

✅ **Data ready:** 1,000 CryoTransformer images converted
✅ **Implementation tested:** All tests pass
✅ **Commands ready:** Copy-paste to run

**Start with quick test:**
```bash
python3 train_ssl_evaluation_fixed.py \
  --image-dir data/cryotransformer/images \
  --coords-file data/cryotransformer/coordinates.json \
  --output-dir experiments/ssl_cryotransformer_quick \
  --labeled-ratios 0.05,0.10 \
  --max-images 1000 \
  --particle-radius 10
```

Good luck with your experiments! 🚀
