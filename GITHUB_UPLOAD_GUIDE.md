# GitHub Upload Guide: Fair Comparison V3

This guide will help you upload the Fair Comparison V3 results (Standard U-Net vs Attention U-Net) to GitHub.

---

## Pre-Upload Checklist

### ✅ Files Ready for Upload

**Documentation**:
- [x] `FAIR_COMPARISON_V3_README.md` - Main project README
- [x] `FAIR_COMPARISON_V3_TECHNICAL_REPORT.md` - Detailed technical report
- [x] `visualize_fair_comparison_v3.py` - Visualization script

**Code**:
- [x] `train_standard_unet_fair_comparison.py` - Standard U-Net training script
- [x] `train_attention_unet_fair_comparison.py` - Attention U-Net training script
- [x] `models/unet.py` - U-Net architecture implementation
- [x] `models/attention_unet.py` - Attention U-Net implementation

**Results**:
- [x] `experiments/fair_comparison_v3/standard_unet/iteration_0_supervised/metrics.json`
- [x] `experiments/fair_comparison_v3/attention_unet/iteration_0_supervised/metrics.json`
- [x] `experiments/fair_comparison_v3/comparison_visualization.png`
- [x] `experiments/fair_comparison_v3/comparison_bar_chart.png`

**Model Checkpoints** (optional - large files):
- [ ] `experiments/fair_comparison_v3/standard_unet/iteration_0_supervised/best_model.pt` (119MB)
- [ ] `experiments/fair_comparison_v3/attention_unet/iteration_0_supervised/best_model.pt` (120MB)

---

## Step 1: Repository Setup

### Option A: Create New Repository

```bash
# Navigate to your project directory
cd /home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU

# Initialize git repository (if not already done)
git init

# Create a new repository on GitHub
# Go to: https://github.com/new
# Repository name: CryoEM-Fair-Comparison-V3
# Description: Fair architectural comparison between Standard U-Net and Attention U-Net for CryoEM particle picking
# Public or Private: Your choice
# Do NOT initialize with README (we already have one)

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/CryoEM-Fair-Comparison-V3.git
```

### Option B: Use Existing Repository

```bash
# If you already have a repository
cd /home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU

# Check current remote
git remote -v

# If needed, add or update remote
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

---

## Step 2: Prepare Files for Upload

### 2.1 Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Data files (too large for GitHub)
data/
*.zarr/
*.npy
*.npz

# Model checkpoints (optional - use Git LFS or external storage)
*.pt
*.pth
*.ckpt

# Jupyter notebooks checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
nohup.out

# OS
.DS_Store
Thumbs.db

# Experiments (keep only fair_comparison_v3)
experiments/*
!experiments/fair_comparison_v3/
experiments/fair_comparison_v3/*/best_model.pt
experiments/fair_comparison_v3/*/model.pt

# Keep important files
!experiments/fair_comparison_v3/**/*.json
!experiments/fair_comparison_v3/**/*.png
!experiments/fair_comparison_v3/**/*.md
EOF
```

### 2.2 Organize Files

Create a clean directory structure:

```bash
# Create a clean directory for GitHub upload
mkdir -p github_upload
cd github_upload

# Copy essential files
cp ../FAIR_COMPARISON_V3_README.md ./README.md
cp ../FAIR_COMPARISON_V3_TECHNICAL_REPORT.md ./TECHNICAL_REPORT.md
cp ../visualize_fair_comparison_v3.py ./visualize_results.py

# Copy training scripts
mkdir -p scripts
cp ../train_standard_unet_fair_comparison.py ./scripts/
cp ../train_attention_unet_fair_comparison.py ./scripts/

# Copy model implementations
mkdir -p models
cp ../models/unet.py ./models/
cp ../models/attention_unet.py ./models/
cp ../models/__init__.py ./models/ 2>/dev/null || true

# Copy results
mkdir -p results
cp -r ../experiments/fair_comparison_v3/standard_unet/iteration_0_supervised/metrics.json \
   ./results/standard_unet_metrics.json
cp -r ../experiments/fair_comparison_v3/attention_unet/iteration_0_supervised/metrics.json \
   ./results/attention_unet_metrics.json
cp -r ../experiments/fair_comparison_v3/*.png ./results/ 2>/dev/null || true
```

### 2.3 Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.23.0
scipy>=1.9.0

# Data handling
zarr>=2.12.0
h5py>=3.7.0

# Visualization
matplotlib>=3.5.2
seaborn>=0.11.2

# Metrics and utilities
scikit-learn>=1.1.1
tqdm>=4.64.0

# Optional: Weights & Biases for experiment tracking
# wandb>=0.13.0
EOF
```

### 2.4 Create environment.yml (for conda)

```bash
cat > environment.yml << 'EOF'
name: cryoem-fair-comparison
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch>=1.12.0
  - torchvision>=0.13.0
  - cudatoolkit=11.3
  - numpy>=1.23.0
  - scipy>=1.9.0
  - matplotlib>=3.5.2
  - seaborn>=0.11.2
  - scikit-learn>=1.1.1
  - zarr>=2.12.0
  - h5py>=3.7.0
  - tqdm>=4.64.0
  - pip
  - pip:
    - wandb  # Optional
EOF
```

---

## Step 3: Git Operations

### 3.1 Stage Files

```bash
# Add all files
git add .

# Check what will be committed
git status

# Expected output should show:
# - README.md
# - TECHNICAL_REPORT.md
# - visualize_results.py
# - scripts/
# - models/
# - results/
# - requirements.txt
# - environment.yml
# - .gitignore
```

### 3.2 Commit Changes

```bash
# Create commit
git commit -m "Add Fair Comparison V3: Standard U-Net vs Attention U-Net

- Comprehensive fair comparison between Standard U-Net and Attention U-Net
- Attention U-Net achieves +0.53% F1 improvement with only +1.13% more parameters
- Includes detailed technical report and reproducible training scripts
- Results: Attention U-Net F1=73.03% vs Standard U-Net F1=72.64%
- All experiments conducted with identical training conditions (fair comparison validated)

Files included:
- README.md: Project overview and results summary
- TECHNICAL_REPORT.md: Detailed technical analysis
- scripts/: Training scripts for both architectures
- models/: Architecture implementations
- results/: Metrics and visualizations
- requirements.txt: Python dependencies
- environment.yml: Conda environment specification"
```

### 3.3 Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main

# If you encounter authentication issues, use personal access token:
# 1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
# 2. Generate new token with 'repo' scope
# 3. Use token as password when prompted
```

---

## Step 4: GitHub Repository Configuration

### 4.1 Add Repository Description

On GitHub, go to your repository and add:

**Description**:
```
Fair architectural comparison: Standard U-Net vs Attention U-Net for CryoEM particle picking. Attention U-Net achieves +0.53% F1 improvement with minimal parameter overhead.
```

**Topics/Tags**:
```
cryoem
deep-learning
u-net
attention-mechanism
particle-picking
biomedical-imaging
pytorch
computer-vision
fair-comparison
segmentation
```

### 4.2 Create GitHub Releases (Optional)

Create a release for your results:

1. Go to **Releases** → **Create a new release**
2. Tag version: `v1.0.0`
3. Release title: `Fair Comparison V3 - Initial Results`
4. Description:
```markdown
## Fair Comparison V3: Standard U-Net vs Attention U-Net

### Results Summary
- **Standard U-Net**: F1 = 72.64% (Epoch 70)
- **Attention U-Net**: F1 = 73.03% (Epoch 80)
- **Improvement**: +0.53% F1, +2.05% Recall
- **Parameter Overhead**: Only +1.13%

### What's Included
- Training scripts for both architectures
- Model implementations
- Validation metrics (JSON format)
- Comparison visualizations
- Comprehensive technical report

### Reproducibility
All experiments conducted with:
- Fixed random seed (42)
- Identical training data (4,653 images)
- Identical validation data (534 images)
- Same loss function, optimizer, and hyperparameters

See README.md for full details and reproduction instructions.
```

5. Attach files (optional):
   - `results/standard_unet_metrics.json`
   - `results/attention_unet_metrics.json`
   - `results/comparison_visualization.png`
   - `results/comparison_bar_chart.png`

---

## Step 5: Handle Large Files (Model Checkpoints)

Model checkpoints are too large for GitHub (>100MB). Options:

### Option A: Git LFS (Recommended)

```bash
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes

# Add model checkpoints
git add experiments/fair_comparison_v3/*/iteration_0_supervised/*.pt

# Commit and push
git commit -m "Add model checkpoints via Git LFS"
git push origin main
```

### Option B: External Storage

Upload to:
- **Zenodo**: https://zenodo.org (permanent DOI, recommended for research)
- **Google Drive**: Share link in README
- **Dropbox**: Share link in README
- **Hugging Face Hub**: https://huggingface.co/models

Add download links to README:

```markdown
## Model Checkpoints

Pre-trained model checkpoints are available for download:

- **Standard U-Net** (119 MB): [Download from Zenodo](https://zenodo.org/...)
- **Attention U-Net** (120 MB): [Download from Zenodo](https://zenodo.org/...)

Place downloaded models in:
```
results/
├── standard_unet_best_model.pt
└── attention_unet_best_model.pt
```
```

### Option C: Skip Model Checkpoints

If you don't need to share model weights:

```bash
# .gitignore already excludes .pt files
# Just share the training scripts and metrics
```

---

## Step 6: Enhance Repository

### 6.1 Add Badges to README

Add these badges at the top of your README:

```markdown
# Fair Comparison V3: Standard U-Net vs Attention U-Net

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/...)

[Rest of README content...]
```

### 6.2 Add LICENSE

Choose a license (MIT recommended for research):

```bash
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

git add LICENSE
git commit -m "Add MIT license"
git push origin main
```

### 6.3 Add CITATION.cff

```bash
cat > CITATION.cff << 'EOF'
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "[Your Last Name]"
    given-names: "[Your First Name]"
    orcid: "https://orcid.org/YOUR-ORCID-ID"
title: "Fair Comparison V3: Standard U-Net vs Attention U-Net for CryoEM Particle Picking"
version: 1.0.0
date-released: 2025-11-05
url: "https://github.com/YOUR_USERNAME/CryoEM-Fair-Comparison-V3"
preferred-citation:
  type: article
  authors:
    - family-names: "[Your Last Name]"
      given-names: "[Your First Name]"
  title: "Fair Architectural Comparison: Standard U-Net vs Attention U-Net for CryoEM Particle Picking"
  year: 2025
  journal: "[Journal Name]"
EOF

git add CITATION.cff
git commit -m "Add citation file"
git push origin main
```

---

## Step 7: Final Verification

### 7.1 Check Repository on GitHub

Visit your repository and verify:

- [ ] README displays correctly
- [ ] All files are present
- [ ] Visualizations render correctly
- [ ] Code syntax highlighting works
- [ ] Links work (if any)
- [ ] License is displayed

### 7.2 Test Reproduction

Clone your repository in a fresh location and test:

```bash
# Clone repository
cd /tmp
git clone https://github.com/YOUR_USERNAME/CryoEM-Fair-Comparison-V3.git
cd CryoEM-Fair-Comparison-V3

# Create environment
conda env create -f environment.yml
conda activate cryoem-fair-comparison

# Install dependencies
pip install -r requirements.txt

# Run visualization
python visualize_results.py

# Expected output:
# - Plots generated successfully
# - No errors
```

---

## Step 8: Share Your Work

### 8.1 Academic Sharing

- [ ] Submit preprint to arXiv
- [ ] Share on Twitter with #CryoEM hashtag
- [ ] Post on ResearchGate
- [ ] Share in relevant Reddit communities (r/MachineLearning, r/CryoEM)
- [ ] Email to collaborators

### 8.2 Update README with Publication Info

Once published, add to README:

```markdown
## Citation

If you use this work, please cite:

```bibtex
@article{yourname2025fair,
  title={Fair Architectural Comparison: Standard U-Net vs Attention U-Net for CryoEM Particle Picking},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025},
  volume={X},
  pages={XXX-XXX},
  doi={10.XXXX/XXXXX}
}
```
```

---

## Troubleshooting

### Issue: "Repository not found"
**Solution**: Check remote URL: `git remote -v`

### Issue: "Permission denied"
**Solution**: Use personal access token instead of password

### Issue: "File too large"
**Solution**: Use Git LFS or external storage for model checkpoints

### Issue: "Merge conflict"
**Solution**: Pull first, then push: `git pull origin main --rebase && git push origin main`

### Issue: "Push rejected"
**Solution**: Force push (use carefully): `git push -f origin main`

---

## Quick Upload Script

Save this as `upload_to_github.sh` for easy upload:

```bash
#!/bin/bash

# Quick GitHub upload script for Fair Comparison V3

echo "Starting GitHub upload..."

# Stage all files
git add .

# Show status
echo "Files to be committed:"
git status

# Prompt for commit message
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update Fair Comparison V3 results"
fi

# Commit
git commit -m "$commit_msg"

# Push
git push origin main

echo "Upload complete! Check your repository at:"
git remote get-url origin
```

Make executable: `chmod +x upload_to_github.sh`

---

## Summary

**You've successfully prepared your Fair Comparison V3 results for GitHub!**

Your repository now includes:
- ✅ Comprehensive README
- ✅ Detailed technical report
- ✅ Reproducible training scripts
- ✅ Model implementations
- ✅ Validation metrics
- ✅ Comparison visualizations
- ✅ Environment specifications
- ✅ Clear documentation

**Next Steps**:
1. Share your repository link
2. Submit to arXiv (if writing paper)
3. Share on social media
4. Apply findings to production pipeline

Good luck with your research! 🚀
