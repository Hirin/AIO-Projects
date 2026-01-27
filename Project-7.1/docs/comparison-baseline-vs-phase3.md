# So Sánh: Baseline BTC vs Phase 3 VideoMAE

> **Tài liệu so sánh chi tiết** giữa hai approaches cho bài toán Video Action Recognition trên HMDB51.

---

## 📊 Tổng Quan So Sánh

| Aspect | Baseline BTC | Phase 3 VideoMAE |
|--------|--------------|------------------|
| **Model** | ViT-Small | VideoMAE-Base |
| **Pretrain Data** | ImageNet (Images) | Kinetics-400 (Videos) |
| **Parameters** | ~22M | ~86M |
| **Training Time** | ~21 phút | ~6.8 giờ |
| **Epochs** | 4 | 40 (30+10) |
| **Expected Accuracy** | ~0.65-0.70 | >0.83 |

---

## 🏗️ Kiến Trúc Model

### Baseline: LightweightViTForAction

```
Input [B, 16, 3, 224, 224]
        │
        ▼
┌─────────────────┐
│   ViT-Small     │  ← ImageNet pretrained
│  (per-frame)    │  ← Xử lý từng frame độc lập
└────────┬────────┘
         │
         ▼
    Mean Pooling      ← Chỉ average, không học temporal
         │
         ▼
    Linear(384, 51)
         │
         ▼
    Output [B, 51]
```

### Phase 3: VideoMAE

```
Input [B, 16, 3, 224, 224]
        │
        ▼
┌─────────────────┐
│    VideoMAE     │  ← Kinetics-400 pretrained
│   Encoder       │  ← Temporal + Spatial Attention
│                 │  ← Masked Autoencoder pretraining
└────────┬────────┘
         │
         ▼
    Built-in Pooling  ← Learned temporal aggregation
         │
         ▼
    Linear(768, 51)
         │
         ▼
    Output [B, 51]
```

### Điểm Khác Biệt Chính

| Component | Baseline | Phase 3 |
|-----------|----------|---------|
| **Backbone** | `vit_small_patch16_224` | `videomae-base-finetuned-kinetics` |
| **Hidden Dim** | 384 | 768 |
| **Temporal Modeling** | Mean Pool (no learning) | Temporal Attention (learned) |
| **Pretrain Task** | Image Classification | Video Masked Autoencoding |

---

## ⚙️ Hyperparameters

| Parameter | Baseline | Phase 3 |
|-----------|----------|---------|
| **Batch Size** | 16 | 8 |
| **Effective Batch** | 64 (accum=4) | 32 (accum=4) |
| **Learning Rate** | 1e-4 (backbone), 5e-4 (head) | 5e-5 (P1), 1e-6 (P2) |
| **Weight Decay** | 0.05 | 0.05 |
| **Epochs** | 4 | 40 (30+10) |
| **Optimizer** | AdamW | AdamW |
| **LR Schedule** | None | Cosine + Warmup |
| **Warmup** | None | 10% |

---

## 📊 Data Augmentation

### Baseline

```python
# Training
├── Random Scale (0.8-1.0)
├── Random Crop → 224x224
├── Random H-Flip (p=0.5)
└── Normalize (mean=0.5, std=0.5)

# ⚠️ KHÔNG consistent across frames
# ⚠️ Mỗi frame có thể crop/flip khác nhau
```

### Phase 3

```python
# Training
├── Resize short edge → 256
├── RandomResizedCrop (scale=0.8-1.0, ratio=0.75-1.33)
├── Random H-Flip (p=0.5)
├── Normalize (VideoMAE mean/std)
└── Mixup (α=0.8, prob=1.0)  # Phase 1 only

# ✅ CONSISTENT across frames
# ✅ Cùng crop params cho tất cả frames trong video
```

### So Sánh Chi Tiết

| Augmentation | Baseline | Phase 3 |
|--------------|----------|---------|
| **Random Crop** | ✅ | ✅ (Consistent) |
| **Horizontal Flip** | ✅ | ✅ (Consistent) |
| **Consistent Transform** | ❌ | ✅ |
| **Mixup** | ❌ | ✅ (Phase 1) |
| **Label Smoothing** | ❌ | ✅ (Phase 2) |
| **Resize Strategy** | Direct to 224 | Short edge 256 → Crop 224 |

---

## 🏋️ Training Strategy

### Baseline: Single-Phase Training

```
Epoch 1 ──► Epoch 2 ──► Epoch 3 ──► Epoch 4 ──► Done
   │           │           │           │
   └───────────┴───────────┴───────────┘
         Constant Strategy Throughout
         
• Fixed LR: 1e-4 / 5e-4
• No augmentation changes
• Cross-Entropy loss
```

### Phase 3: 2-Stage Training

```
┌─────────────────────────────────────┐
│           PHASE 1 (30 epochs)       │
│  • Mixup ON (α=0.8)                 │
│  • Higher LR (5e-5)                 │
│  • Cosine Schedule + Warmup         │
│  • Soft Cross-Entropy Loss          │
│  → Learn robust, diverse features   │
└──────────────────┬──────────────────┘
                   │
                   ▼ Load Best P1 Model
┌─────────────────────────────────────┐
│           PHASE 2 (10 epochs)       │
│  • Mixup OFF                        │
│  • Label Smoothing (0.1)            │
│  • Very Low LR (1e-6)               │
│  • Standard Cross-Entropy           │
│  → Polish without overfitting       │
└─────────────────────────────────────┘
```

---

## 🔍 Inference

### Baseline: Single-View

```
Video ──► Resize 224 ──► Model ──► Prediction
                │
           1 forward pass
```

### Phase 3: 6-View TTA

```
Video ──┬──► Center Crop ───────┬───► Model ──┐
        ├──► Left/Top Crop ─────┤             │
        ├──► Right/Bottom Crop ─┤             │
        ├──► Center + Flip ─────┤             ├──► Average ──► Prediction
        ├──► Left/Top + Flip ───┤             │
        └──► Right/Bottom + Flip┘             │
                │                             │
           6 forward passes ────────────────────
```

### Comparison

| Aspect | Baseline | Phase 3 |
|--------|----------|---------|
| **Views** | 1 | 6 |
| **Crops** | Center only | Center + Side crops |
| **Flips** | No | Yes (all crops) |
| **Aggregation** | N/A | Average logits |
| **Robustness** | Low | High |
| **Inference Time** | 1x | ~6x |

---

## 📈 Kỹ Thuật Regularization

| Technique | Baseline | Phase 3 | Mục Đích |
|-----------|----------|---------|----------|
| **Mixup** | ❌ | ✅ | Smooth decision boundaries |
| **Label Smoothing** | ❌ | ✅ | Prevent overconfidence |
| **Weight Decay** | ✅ 0.05 | ✅ 0.05 | L2 regularization |
| **Gradient Clipping** | ❌ | ✅ max=1.0 | Training stability |
| **Dropout** | ❌ | Built-in | Prevent overfitting |

---

## 💻 Code Comparison

### Model Loading

**Baseline:**
```python
import timm
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
head = nn.Linear(384, 51)
```

**Phase 3:**
```python
from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
    num_frames=16
)
```

### Loss Function

**Baseline:**
```python
loss = F.cross_entropy(logits, labels)
```

**Phase 3:**
```python
# Phase 1: Soft CE for Mixup
if use_mixup:
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(targets * log_probs, dim=1).mean()
# Phase 2: CE with Label Smoothing
else:
    loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
```

### Data Transform

**Baseline:**
```python
class VideoTransform:
    def __call__(self, frames):
        # Each frame transformed independently
        for frame in frames:
            frame = random_crop(frame)  # Different for each frame!
            frame = maybe_flip(frame)   # Different for each frame!
```

**Phase 3:**
```python
class VideoDataset:
    def __getitem__(self, idx):
        # Get params ONCE for entire video
        i, j, h, w = RandomResizedCrop.get_params(frames[0], ...)
        is_flip = random.random() > 0.5
        
        # Apply SAME transform to ALL frames
        for img in frames:
            img = TF.resized_crop(img, i, j, h, w, ...)
            if is_flip:
                img = TF.hflip(img)
```

---

## 📊 Expected Performance

```
                    Accuracy
    0.5   0.6   0.7   0.8   0.9   1.0
     │     │     │     │     │     │
     ├─────┴─────┼─────┴─────┼─────┤
     │           │           │     │
     │  Baseline │           │     │
     │  (0.65-0.70)          │     │
     │           │           │     │
     │           │   Phase 3 │     │
     │           │   (>0.83) │     │
     │           │           │     │
     └───────────┴───────────┴─────┘
```

---

## 🎯 Khi Nào Dùng Gì?

### Dùng Baseline Khi:
- ✅ Prototype nhanh, test pipeline
- ✅ Limited compute resources
- ✅ Baseline comparison
- ✅ Học về video classification cơ bản

### Dùng Phase 3 Khi:
- ✅ Cần accuracy cao
- ✅ Có đủ GPU time (~7 giờ)
- ✅ Competition/Production
- ✅ Dataset video action recognition

---

## 📝 Summary

| Aspect | Winner | Reason |
|--------|--------|--------|
| **Accuracy** | Phase 3 | VideoMAE + advanced techniques |
| **Speed** | Baseline | 20x faster training |
| **Simplicity** | Baseline | Fewer components |
| **Robustness** | Phase 3 | Mixup + TTA + Label Smoothing |
| **Temporal Learning** | Phase 3 | Temporal attention vs mean pool |
| **Resource Efficiency** | Baseline | Smaller model, less memory |

---

## 🔄 Improvement Path

```
Baseline BTC
     │
     ├──► Add Consistent Transforms
     │
     ├──► Increase Epochs (4 → 20)
     │
     ├──► Add Learning Rate Schedule
     │
     ├──► Add Mixup
     │
     ├──► Switch to Video Pretrained Model
     │
     ├──► Add 2-Stage Training
     │
     ├──► Add Multi-View TTA
     │
     ▼
Phase 3 VideoMAE
```
