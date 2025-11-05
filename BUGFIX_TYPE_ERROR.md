# Bug Fix: Type Conversion Error

## Problem

The training script crashed with the following error:
```
RuntimeError: result type Float can't be cast to the desired output type Long
```

## Root Cause

The dataset returns masks as `int8` (or `int64`) tensors, but PyTorch's `binary_cross_entropy_with_logits` expects float tensors. The loss functions were not converting the target tensors to float before computation.

## Solution

Added `.float()` conversion to targets in all loss functions:

1. **FocalLoss** (line 54 in `improved_losses.py`)
2. **DiceLoss** (line 137 in `improved_losses.py`)
3. **TverskyLoss** (line 234 in `improved_losses.py`)

### Code Change:
```python
# Before (line 50-52):
if targets.dim() == 3:
    targets = targets.unsqueeze(1)

# After (line 50-54):
if targets.dim() == 3:
    targets = targets.unsqueeze(1)

# Convert targets to float (important for loss computation)
targets = targets.float()
```

## Verification

✅ **Syntax Check**: Passed
✅ **Unit Test (normal targets)**: Passed
✅ **Unit Test (with ignore masks)**: Passed

### Test Results:
```
Loss computation with int8 targets: SUCCESS
- Loss: 0.2717
- Focal: 0.1743
- Dice: 0.4988
- Target dtype: torch.int8 → converted to float internally

Loss computation with ignore masks (-1): SUCCESS
- Loss: 0.2698
- Focal: 0.1714
- Dice: 0.4995
- Valid pixels: 5416 / 8192 (ignores -1 correctly)
```

## Impact

- ✅ Training can now proceed without type errors
- ✅ All loss functions handle int8/int64 target tensors
- ✅ Ignore masks (-1 values) are properly handled
- ✅ No change to loss computation logic, just type safety

## Training Ready

The script is now ready to run:
```bash
python train_unet_selftraining_improved.py \
    --image-dir data/unet_full_train/images \
    --coords-file data/unet_full_train/coordinates.json \
    --output-dir experiments/unet_improved_v1 \
    --initial-epochs 100 \
    --self-training-iterations 3 \
    --retrain-epochs 30 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --device cuda:0 \
    --multi-gpu
```

## Files Modified

- `improved_losses.py`: Added `.float()` conversion in FocalLoss, DiceLoss, and TverskyLoss

---

**Status**: ✅ FIXED - Ready for production training
