# Final Results Summary Table

## Ablation Study Results

| Experiment | Status | Best Val AUC | Test AUC | Test AP | Precision | Recall | F1 Score | Training Notes |
|------------|--------|-------------|----------|---------|-----------|---------|----------|----------------|
| **PU-only** | ✅ Complete | **0.951** | **0.936** | **0.984** | 0.967 | **0.787** | **0.868** | Early stop @ epoch 20 |
| **PUFP** | ✅ Complete | 0.925 | 0.913 | 0.939 | **1.000** | 0.547 | 0.707 | Early stop @ epoch 38 |
| **Selective FixMatch** | ✅ Complete | 0.907 | TBD | TBD | TBD | TBD | TBD | Full 100 epochs |

## Key Findings

1. **🏆 PU-only is the winner** - Best overall performance across most metrics
2. **📊 2.3% higher Test AUC** than PUFP approach
3. **🎯 24% better Recall** - Critical for particle detection
4. **⚡ Faster convergence** - Early stopping at epoch 20 vs 38
5. **🔬 Clean training signal** - No false positive confusion

## Recommended Model

**Primary**: PU-only learning
- Model: `/home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU/experiments/ablation_PU_final/best_model.pt`
- Config: `/home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU/experiments/ablation_PU_final/config.yaml`
- Test AUC: **0.936**
- Precision: **0.967**
- Recall: **0.787**

**Alternative**: Selective FixMatch for extended training scenarios
- Model: `/home/uuni/cryoppp/fixmatch/CryoEM_FixMatch_PU/experiments/selective_fixmatch_final/best_model.pt`
- Best Val AUC: **0.907**
- Training: Stable over 100 epochs