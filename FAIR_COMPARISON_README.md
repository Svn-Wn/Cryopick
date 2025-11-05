# Fair Comparison: Attention U-Net vs Standard U-Net

## 🎯 Objective

Conduct a **rigorous, scientifically sound comparison** between Attention U-Net and Standard U-Net architectures for CryoEM particle picking under **completely identical training conditions**.

---

## 📋 Quick Start

### Step 1: Review the Protocol
```bash
cat FAIR_COMPARISON_PROTOCOL.md
```

This document contains all training settings, fairness criteria, and expected outcomes.

### Step 2: Train Standard U-Net
```bash
./run_fair_comparison_standard.sh
```

**Time:** ~15 hours on 2× RTX A6000
**Output:** `experiments/fair_comparison/standard_unet/`

### Step 3: Train Attention U-Net
```bash
./run_fair_comparison_attention.sh
```

**Time:** ~15 hours on 2× RTX A6000
**Output:** `experiments/fair_comparison/attention_unet/`

### Step 4: Compare Results
```bash
python3 compare_fair_results.py
```

This will show a detailed comparison table and determine the winner.

---

## 🔬 Scientific Rigor

### What Makes This Fair?

#### ✅ Identical Data
- **Training**: 5,172 images (768×768×3)
- **Validation**: 534 images (768×768×3)
- **Same split**: Both models see exact same images

#### ✅ Identical Training
- **Epochs**: 100
- **Batch size**: 16 (split across 2 GPUs)
- **Learning rate**: 0.001
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Loss**: CombinedLoss (Focal + Dice)
- **Random seed**: 42 (fixed)

#### ✅ Identical Hardware
- **GPUs**: 2× NVIDIA RTX A6000 (48GB each)
- **Mode**: DataParallel (splits batch across GPUs)
- **CUDA**: Deterministic mode enabled

#### ✅ Identical Evaluation
- Both evaluated on **same validation set**
- Same metrics: Precision, Recall, F1, IoU, AUC
- No cherry-picking of checkpoints

---

## 📊 Expected Training Time

| Phase | Time per Epoch | Total Time |
|-------|---------------|------------|
| Training (100 epochs) | ~9 minutes | ~15 hours |
| Validation | ~30 seconds | Included |

**Total for both models**: ~30 hours (if run sequentially)

**Recommendation**: Train one model, then the other. Or use separate GPU assignments if running in parallel.

---

## 🖥️ Multi-GPU Configuration

Both training scripts use **PyTorch DataParallel**:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

This automatically:
- Splits each batch across available GPUs
- Aggregates gradients for weight updates
- Uses both A6000 GPUs efficiently

**Memory per GPU**: ~17-20GB (fits comfortably in 48GB)

---

## 📂 Output Structure

```
experiments/fair_comparison/
├── FAIR_COMPARISON_PROTOCOL.md       # Protocol document
├── standard_unet/
│   ├── training.log                   # Full training log
│   └── iteration_0_supervised/
│       ├── best_model.pt              # Best checkpoint
│       ├── model.pt                   # Final model
│       ├── metrics.json               # Final metrics ⭐
│       └── visualization.png          # Predictions
│
├── attention_unet/
│   ├── training.log
│   └── iteration_0_supervised/
│       ├── best_model.pt
│       ├── model.pt
│       ├── metrics.json               # Final metrics ⭐
│       └── visualization.png
│
└── comparison_results.json            # Side-by-side comparison ⭐
```

---

## 🔍 Monitoring Training

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Training Progress
```bash
# Standard U-Net
tail -f experiments/fair_comparison/standard_unet/training.log

# Attention U-Net
tail -f experiments/fair_comparison/attention_unet/training.log
```

### Check Current Epoch
```bash
# Standard U-Net
grep "Epoch" experiments/fair_comparison/standard_unet/training.log | tail -1

# Attention U-Net
grep "Epoch" experiments/fair_comparison/attention_unet/training.log | tail -1
```

---

## 📈 Results Interpretation

After running `compare_fair_results.py`, you'll see:

```
────────────────────────────────────────────────────────────────────────────────
Metric               Standard U-Net       Attention U-Net      Δ (Difference)
────────────────────────────────────────────────────────────────────────────────
Precision            XX.XX%               XX.XX%               +X.XX% ✅
Recall               XX.XX%               XX.XX%               -X.XX% ⚠️
F1 Score             XX.XX%               XX.XX%               +X.XX% ✅
IoU                  X.XXXX               X.XXXX               +X.XXXX ✅
AUC                  X.XXXX               X.XXXX               +X.XXXX ✅
────────────────────────────────────────────────────────────────────────────────
```

### Winner Criteria
- **Primary metric**: F1 Score
- **Meaningful difference**: >2% improvement
- **Winner**: Model with higher F1 score

### Expected Outcome
Based on prior experiments, **Attention U-Net** is expected to show:
- ✅ **Higher Precision** (+5-10%): Fewer false positives
- ⚠️ **Lower Recall** (-2-5%): May miss some particles
- ✅ **Higher F1** (+2-5%): Better overall balance
- ✅ **Higher IoU** (+3-8%): Better segmentation quality

---

## 🐛 Troubleshooting

### Out of Memory
**Symptom**: `torch.OutOfMemoryError`

**Solution**: Reduce batch size
```bash
# Edit the training script
BATCH_SIZE=8  # Instead of 16
```

### Training Crashes
**Symptom**: Process dies without error

**Solutions**:
1. Check GPU status: `nvidia-smi`
2. Check system logs: `dmesg | tail -50`
3. Verify data: `ls data/cryotransformer_preprocessed/train/*.npz | wc -l`

### Different Number of Samples
**Symptom**: Models report different training sizes

**Solution**: Ensure both load data the same way
```bash
# Check data loading
python3 -c "
import glob
train = glob.glob('data/cryotransformer_preprocessed/train/*.npz')
val = glob.glob('data/cryotransformer_preprocessed/val/*.npz')
print(f'Train: {len(train)} files')
print(f'Val: {len(val)} files')
"
```

---

## 📝 For Your Paper

### Methods Section

> **Model Comparison**: We compared Standard U-Net and Attention U-Net architectures on the CryoTransformer dataset (5,172 training images, 534 validation images). Both models were trained for 100 epochs with identical hyperparameters (batch size=16, learning rate=0.001, Adam optimizer) using a combined Focal+Dice loss. Training was performed on 2× NVIDIA RTX A6000 GPUs using PyTorch DataParallel. Random seeds were fixed (seed=42) for reproducibility. Models were evaluated on precision, recall, F1 score, IoU, and AUC.

### Results Table

| Model | Parameters | Precision | Recall | F1 Score | IoU | AUC |
|-------|-----------|-----------|--------|----------|-----|-----|
| Standard U-Net | 31.0M | XX.X% | XX.X% | XX.X% | 0.XXX | 0.XXX |
| Attention U-Net | 31.4M | XX.X% | XX.X% | XX.X% | 0.XXX | 0.XXX |

> Attention U-Net achieved X.X% higher F1 score than Standard U-Net, demonstrating the effectiveness of attention mechanisms for particle detection in low-SNR cryo-EM micrographs.

---

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [ ] Data directories exist and contain images
  ```bash
  ls data/cryotransformer_preprocessed/train/*.npz | wc -l  # Should be 52
  ls data/cryotransformer_preprocessed/val/*.npz | wc -l    # Should be 6
  ```

- [ ] GPUs are available and working
  ```bash
  nvidia-smi  # Should show 2× RTX A6000
  ```

- [ ] Conda environment is activated
  ```bash
  conda activate cryopick
  python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
  ```

- [ ] Training scripts exist and are executable
  ```bash
  ls -l run_fair_comparison_*.sh  # Should show executable (-rwxr-xr-x)
  ```

- [ ] Output directory is clean
  ```bash
  rm -rf experiments/fair_comparison  # Clean start (optional)
  ```

---

## 🎓 Citation

If you use this comparison methodology in your research:

```bibtex
@misc{fair_unet_comparison,
  title={Fair Comparison of Attention U-Net and Standard U-Net for CryoEM Particle Picking},
  author={Your Name},
  year={2025},
  note={Rigorous comparison under identical training conditions}
}
```

---

## 📧 Questions?

For issues with:
- **Training scripts**: Check training logs in `experiments/fair_comparison/*/training.log`
- **Data loading**: Verify data exists and is readable
- **GPU problems**: Check `nvidia-smi` and system logs
- **Results interpretation**: Run `compare_fair_results.py` for detailed analysis

---

## 🚀 Ready to Start?

1. **Read the protocol**: `cat FAIR_COMPARISON_PROTOCOL.md`
2. **Run pre-flight checklist** (above)
3. **Start Standard U-Net**: `./run_fair_comparison_standard.sh`
4. **Wait ~15 hours**
5. **Start Attention U-Net**: `./run_fair_comparison_attention.sh`
6. **Wait ~15 hours**
7. **Compare results**: `python3 compare_fair_results.py`
8. **Publish your findings!** 🎉

Good luck with your research!
