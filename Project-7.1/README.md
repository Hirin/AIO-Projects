# Video Action Recognition - Ablation Study

## 📋 Tổng quan

Project thực hiện **Video Action Recognition** trên tập dữ liệu 51 action classes sử dụng các model:
- **ViT-Small/Base** (ImageNet pretrained, frame-level)
- **VideoMAE** (Kinetics pretrained, video-level)

## 📊 Kết quả

| Model | Configuration | Test Accuracy |
|-------|---------------|---------------|
| ViT-Small Baseline | 16 frames, 10 epochs | 69.22% |
| ViT-Base Baseline | 16 frames, 10 epochs | 73.73% |
| VideoMAE 8-Frame | 8 frames, 60 epochs | 83.00% |
| VideoMAE Baseline | 16 frames, 10 epochs | 83.92% |
| VideoMAE + Data Balance | + Focal Loss | 84.00% |
| VideoMAE + Layer Decay | + Mixup | 84.51% |
| **VideoMAE Phase 3** | Full pipeline | **85.10%** |

### Best Configuration (Phase 3)
```python
MODEL_TYPE = "videomae"
NUM_FRAMES = 16
USE_CONSISTENT_SPATIAL_AUG = True
USE_MIXUP = True
USE_LABEL_SMOOTHING = True
USE_TWO_PHASE = True
USE_FLIP_TTA = True
EPOCHS_P1 = 30
EPOCHS_P2 = 10
```

## 📁 Cấu trúc Project

```
Project-7.1/
├── README.md
├── notebooks/
│   ├── unified-ablation.ipynb      # Main notebook với toggle system
│   ├── baseline-btc.ipynb          # Baseline experiments
│   └── project-7-1-phase-3.ipynb   # Phase 3 best result
├── scripts/
│   └── create_unified_notebook.py  # Generator script
├── docs/
│   ├── ablation_study_conclusion.md
│   ├── baseline-btc-analysis.md
│   ├── comparison-baseline-vs-phase3.md
│   └── phase3-videomae-analysis.md
├── results/                    # Training results & visualizations
└── data/                       # Dataset (not tracked)
```

## 🚀 Sử dụng

### 1. Chạy Ablation Study
Upload `unified-ablation.ipynb` lên Kaggle và set các toggles theo configuration mong muốn:

```python
# Quick test all pipelines
WARMUP = True

# Or run specific experiment
WARMUP = False
MODEL_TYPE = "videomae"
USE_MIXUP = True
USE_TWO_PHASE = True
# ...
```

### 2. Regenerate Notebook
```bash
python scripts/create_unified_notebook.py
```

## 🔧 Configuration Toggles

| Toggle | Mô tả | Default |
|--------|-------|---------|
| `WARMUP` | Test pipeline nhanh (5 batches/phase) | False |
| `MODEL_TYPE` | `vit_small` / `vit_base` / `videomae` | videomae |
| `NUM_FRAMES` | Số frames per video | 16 |
| `TRAIN_VAL_RATIO` | Tỷ lệ train/val split | 0.9 |
| `USE_TEST_LABELS` | Download test labels để tính accuracy | True |
| `USE_CONSISTENT_SPATIAL_AUG` | Same crop/flip cho all frames | True |
| `USE_MIXUP` | Mixup augmentation (α=0.8) | True |
| `USE_FOCAL_LOSS` | Focal Loss cho imbalanced data | False |
| `USE_LABEL_SMOOTHING` | Label smoothing (ε=0.1) | True |
| `USE_TWO_PHASE` | 2-Phase: Mixup → Label Smoothing | True |
| `USE_LAYER_DECAY` | Layer-wise LR decay | False |
| `USE_FLIP_TTA` | 6-view TTA (3 crops × 2 flip) | True |

## 📈 Training Pipeline

### Two-Phase Training
1. **Phase 1** (30 epochs): Mixup + High LR (5e-5)
2. **Phase 2** (10 epochs): Label Smoothing + Low LR (1e-6)

### Data Augmentation
- Consistent spatial augmentation (same crop/flip across frames)
- Mixup với α=0.8
- 6-view TTA at inference

## 📚 Dependencies

```
torch >= 2.0
torchvision
timm
transformers
scikit-learn
pandas
matplotlib
tqdm
```

## 📝 License

MIT License
