# GitHub Setup Guide 🚀

Quick guide to upload your U-Net Self-Training project to GitHub.

---

## 📋 Preparation Checklist

### Step 1: Files to Include

✅ **Code Files** (include):
```
├── inference_standalone.py         # Inference script
├── train_unet_selftraining_improved.py  # Training script
├── improved_losses.py              # Loss functions
├── improved_augmentation.py        # Augmentation
├── requirements.txt                # Dependencies
├── requirements_inference.txt      # Minimal dependencies
└── .gitignore                      # Git ignore file
```

✅ **Documentation** (include):
```
├── README_UNET.md                  # Main readme (create this)
├── TRAINING_RESULTS_SUMMARY.md     # Results
├── MODEL_DEPLOYMENT_GUIDE.md       # Deployment guide
├── DEPLOYMENT_QUICKSTART.md        # Quick start
├── IMPROVEMENTS_SUMMARY.md         # Technical details
└── LICENSE                         # License file
```

❌ **Large Files** (exclude or use Git LFS):
```
├── experiments/                    # 119 MB models
├── data/                           # Dataset (gigabytes)
├── __pycache__/                    # Python cache
└── *.log                           # Log files
```

---

## 🔧 Step 2: Handle Large Model Files

**Problem**: GitHub has 100 MB file limit, your model.pt is 119 MB

**Solution Options**:

### Option A: Git LFS (Recommended if <1GB total)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git add .gitattributes

# Add models
git add experiments/unet_improved_v1/iteration_1_selftrain/model.pt
git commit -m "Add trained models"
```

### Option B: External Hosting (Recommended for >1GB)
Upload to:
- **Google Drive** / **Dropbox** (easiest)
- **Hugging Face** (recommended for ML models)
- **Zenodo** (for academic papers - gets DOI)
- **Your university server**

Then update README with download links.

### Option C: Don't Include Models (minimal repo)
Just include code and documentation, provide instructions to train.

---

## 📝 Step 3: Create .gitignore

Save this as `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv/

# PyTorch
*.pth
*.pt
# Except the final trained models (if using Git LFS)
!experiments/unet_improved_v1/iteration_1_selftrain/model.pt

# Data
data/
*.mrc
*.rec
*.st
*.ali

# Logs
*.log
*.out
*.err
logs/
tensorboard_logs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Training outputs (too large)
experiments/*/iteration_*/
!experiments/*/iteration_*/metrics.json

# Temporary files
tmp/
temp/
*.tmp
```

---

## 🚀 Step 4: Initialize Git Repository

```bash
cd /home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU

# Initialize repo
git init

# Add .gitignore
git add .gitignore

# Add code files
git add inference_standalone.py
git add train_unet_selftraining_improved.py
git add improved_losses.py
git add improved_augmentation.py
git add requirements*.txt

# Add documentation
git add *.md
git add experiments/unet_improved_v1/iteration_*/metrics.json

# Initial commit
git commit -m "Initial commit: U-Net self-training for cryo-EM"
```

---

## 📤 Step 5: Push to GitHub

```bash
# Create repo on GitHub first (via web interface)
# Then link it:

git remote add origin https://github.com/YOUR_USERNAME/CryoEM-UNet-SelfTraining.git
git branch -M main
git push -u origin main
```

---

## 📁 Recommended Repository Structure

```
CryoEM-UNet-SelfTraining/
├── .gitignore
├── LICENSE
├── README.md                        # Main README
├── requirements.txt
├── requirements_inference.txt
│
├── src/                             # Source code
│   ├── inference_standalone.py
│   ├── train_unet_selftraining_improved.py
│   ├── improved_losses.py
│   └── improved_augmentation.py
│
├── docs/                            # Documentation
│   ├── TRAINING_RESULTS_SUMMARY.md
│   ├── MODEL_DEPLOYMENT_GUIDE.md
│   ├── DEPLOYMENT_QUICKSTART.md
│   └── IMPROVEMENTS_SUMMARY.md
│
├── experiments/                     # Results (only metrics)
│   └── unet_improved_v1/
│       ├── iteration_0_supervised/
│       │   └── metrics.json
│       ├── iteration_1_selftrain/
│       │   └── metrics.json
│       └── iteration_2_selftrain/
│           └── metrics.json
│
└── assets/                          # Images for README
    └── visualization_example.png
```

---

## 🏷️ Step 6: Add Model Download Links

If using external hosting, add to README:

```markdown
## 📥 Pretrained Models

Download trained models:

| Model | F1 Score | Download |
|-------|----------|----------|
| Iteration 1 (Best) | 75.95% | [Download (119 MB)](https://drive.google.com/YOUR_LINK) |
| Iteration 0 | 75.87% | [Download (119 MB)](https://drive.google.com/YOUR_LINK) |

After downloading, place in:
\`\`\`
experiments/unet_improved_v1/iteration_1_selftrain/model.pt
\`\`\`
```

---

## ✅ Quick Commands

### Complete GitHub Setup:

```bash
# 1. Create .gitignore (copy from above)
vim .gitignore

# 2. Initialize repo
git init
git add .gitignore
git add *.py *.md requirements*.txt
git add experiments/unet_improved_v1/iteration_*/metrics.json

# 3. Commit
git commit -m "Initial commit: U-Net self-training (75.95% F1)"

# 4. Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### With Git LFS (if including models):

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "experiments/**/*.pt"
git add .gitattributes

# Add everything
git add .
git commit -m "Add trained models via Git LFS"
git push
```

---

## 📊 Repository Badges

Add to your README for a professional look:

```markdown
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/REPO_NAME)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/REPO_NAME)
```

---

## 🔒 What NOT to Upload

❌ **DO NOT upload**:
- Raw dataset (gigabytes)
- Training logs (can be huge)
- Intermediate checkpoints (only keep best models)
- Personal credentials or API keys
- Proprietary data

✅ **DO upload**:
- Source code
- Documentation
- Requirements files
- Final trained models (via LFS or external link)
- Example visualizations
- Metrics/results (JSON files)

---

## 🎯 Recommended Workflow

### Option 1: Minimal Repo (Fastest)
```bash
# Upload only code + docs
git add *.py *.md requirements*.txt
git commit -m "Add code and documentation"
git push

# Provide model download instructions in README
```

### Option 2: Full Repo with Git LFS
```bash
# Upload everything including models
git lfs track "*.pt"
git add .
git commit -m "Complete project with trained models"
git push
```

### Option 3: Code + External Models (Recommended)
```bash
# Upload code + docs
git add *.py *.md requirements*.txt docs/
git commit -m "Add source code and documentation"
git push

# Upload models to Google Drive/Hugging Face
# Add download links to README
```

---

## 📞 Next Steps

1. ✅ Create .gitignore
2. ✅ Initialize git repository
3. ✅ Create GitHub repo (web interface)
4. ✅ Push code
5. ✅ Upload models (Git LFS or external)
6. ✅ Update README with download links
7. ✅ Add LICENSE file
8. ✅ Add badges and screenshots

---

**You're ready to share your research with the world!** 🌟

See the files I created:
- README_UNET.md (rename to README.md after pushing)
- .gitignore
- This guide

Just follow the steps above!
