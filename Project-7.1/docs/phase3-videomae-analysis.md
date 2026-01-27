# Phase 3 - VideoMAE Advanced Training Analysis

> **Tài liệu phân tích chi tiết** về notebook `project-7-1-phase-3.ipynb` - Giải pháp nâng cao sử dụng VideoMAE cho bài toán Video Action Recognition.

---

## 📋 Tổng Quan

| Thông tin | Chi tiết |
|-----------|----------|
| **Bài toán** | Video Action Recognition (Nhận dạng hành động trong video) |
| **Dataset** | HMDB51 (51 classes) |
| **Model** | VideoMAE-Base (Kinetics-400 pretrained) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Platform** | Kaggle (Tesla T4 GPU) |
| **Training Time** | ~6.8 giờ (40 epochs total) |
| **Target Accuracy** | > 0.83 |

---

## 🎯 Các Cải Tiến Chính

1. **Model**: `VideoMAE-base` - SOTA cho dataset video nhỏ
2. **Augmentation**: `Mixup` + Consistent Spatial Transform
3. **Training Strategy**: **2-Stage Fine-tuning**
4. **Inference**: **6-View TTA** (Test Time Augmentation)

---

## 🏗️ Kiến Trúc Model

### VideoMAE Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO                               │
│                  [B, T, C, H, W]                             │
│            (Batch, 16 Frames, 3, 224, 224)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              VideoMAE Encoder                                │
│     (MCG-NJU/videomae-base-finetuned-kinetics)              │
│                                                              │
│    • Pretrained on Kinetics-400 (video dataset)             │
│    • Temporal + Spatial Attention                           │
│    • Masked Autoencoder pretraining                         │
│    • Hidden Dim: 768                                        │
│    • Heads: 12                                              │
│    • Layers: 12                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              VIDEO FEATURES                                  │
│         [B, sequence_length, hidden_dim]                     │
│                                                              │
│    Temporal attention learns motion patterns                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            CLASSIFICATION HEAD                               │
│              nn.Linear(768, 51)                              │
│                                                              │
│                 [B, num_classes]                             │
│                   (B, 51)                                    │
└─────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Specification |
|-----------|---------------|
| **Checkpoint** | `MCG-NJU/videomae-base-finetuned-kinetics` |
| **Pretrain Dataset** | Kinetics-400 (video action recognition) |
| **Hidden Dim** | 768 |
| **Attention Heads** | 12 |
| **Transformer Layers** | 12 |
| **Input Frames** | 16 |
| **Input Size** | 224×224 |

---

## ⚙️ Configuration & Hyperparameters

### Model Config

```python
MODEL_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
IMG_SIZE = 224
RESIZE_SIZE = 256  # Resize short edge trước khi crop
```

### Phase 1 Config (Heavy Augmentation)

```python
EPOCHS_P1 = 30
LR_P1 = 5e-5          # Higher LR cho exploration
MIXUP_ALPHA = 0.8     # Mixup strength
MIXUP_PROB = 1.0      # Always apply mixup
```

### Phase 2 Config (Fine-tuning / Polishing)

```python
EPOCHS_P2 = 10
LR_P2 = 1e-6          # Very low LR để polish
LABEL_SMOOTHING = 0.1 # Prevent overconfidence
```

### Common Config

```python
BATCH_SIZE = 8
ACCUM_STEPS = 4       # Effective batch = 32
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
```

---

## 📊 Data Pipeline

### 1. Frame Sampling Strategy

```python
def __getitem__(self, idx):
    # Training: Random stride cho diversity
    if self.is_train:
        max_stride = max(1, (total_frames - 1) // (self.num_frames - 1))
        stride = random.randint(1, min(max_stride, 4))
    else:
        # Validation: Fixed stride
        stride = max(1, (total_frames - 1) // (self.num_frames - 1))
    
    # Uniform sampling
    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
```

### 2. Consistent Spatial Transform (Key Innovation)

```python
# CRITICAL: Tất cả frames trong 1 video phải có CÙNG transform parameters
if self.is_train:
    # 1. Lấy random parameters MỘT LẦN cho cả video
    i, j, h, w = T.RandomResizedCrop.get_params(
        frames[0], 
        scale=(0.8, 1.0), 
        ratio=(0.75, 1.33)
    )
    is_flip = random.random() > 0.5
    
    # 2. Apply CÙNG parameters cho TẤT CẢ frames
    for img in frames:
        img = TF.resized_crop(img, i, j, h, w, size=(224, 224))
        if is_flip:
            img = TF.hflip(img)
        img = TF.normalize(TF.to_tensor(img), mean=MEAN, std=STD)
```

> 💡 **Tại sao Consistent Transform quan trọng?**
> - Video có temporal coherence - các frames liên tiếp phải nhất quán
> - Inconsistent transform có thể phá vỡ motion patterns
> - Giúp model học được chuyển động thực sự, không bị nhiễu bởi artifacts

### 3. Mixup Augmentation

```python
class MixupCollate:
    def __init__(self, num_classes, alpha=0.8, prob=1.0):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return inputs, F.one_hot(targets, num_classes).float()
        
        # Mix two samples
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size)
        
        # Mixed inputs
        inputs = lam * inputs + (1 - lam) * inputs[index, :]
        
        # Soft labels
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index, :]
        
        return inputs, targets
```

---

## 🏋️ 2-Stage Training Strategy

### Phase 1: Heavy Augmentation (30 Epochs)

```python
print("STARTING PHASE 1 (Mixup Enabled, LR=5e-5, Epochs=30)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=0.05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

for epoch in range(EPOCHS_P1):
    loss, acc = train_epoch(
        model, train_loader_p1, optimizer, scheduler, scaler, 
        DEVICE, ACCUM_STEPS, 
        use_mixup=True  # Mixup ON
    )
```

**Mục đích Phase 1:**
- Học robust features với strong augmentation
- Higher LR cho faster exploration
- Mixup ngăn overfitting sớm

### Phase 2: Fine-tuning (10 Epochs)

```python
print("STARTING PHASE 2 (No Mixup, Label Smooth=0.1, Low LR=1e-6)")

# Load best model từ Phase 1
model = VideoMAEForVideoClassification.from_pretrained("./videomae_phase1_best")

# Very low LR để không phá vỡ learned features
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=0.05)

for epoch in range(EPOCHS_P2):
    loss, acc = train_epoch(
        model, train_loader_p2, optimizer, scheduler, scaler,
        DEVICE, ACCUM_STEPS,
        use_mixup=False,           # Mixup OFF
        label_smoothing=0.1        # Label Smoothing ON
    )
```

**Mục đích Phase 2:**
- Polish model với clean data (no mixup)
- Label Smoothing ngăn overconfident predictions
- Very low LR để fine-tune nhẹ

### Training Loop với Dual Loss Support

```python
def train_epoch(..., use_mixup=True, label_smoothing=0.0):
    for inputs, targets in loader:
        outputs = model(inputs)
        logits = outputs.logits
        
        if use_mixup:
            # Soft cross-entropy cho mixup labels
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.sum(targets * log_probs, dim=1).mean()
        else:
            # Standard CE với optional label smoothing
            loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        
        # Gradient accumulation + AMP
        loss = loss / accum_steps
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
```

---

## 🔍 6-View Test Time Augmentation (TTA)

### MultiViewTestDataset

```python
class MultiViewTestDataset(Dataset):
    """Tạo 6 góc nhìn cho mỗi video."""
    
    def __getitem__(self, idx):
        # 1. Resize shortest side to 256
        frames = [TF.resize(img, 256) for img in frames]
        
        views = []
        w, h = frames[0].size
        crop_size = 224
        
        # --- 3 STANDARD CROPS ---
        # Center Crop
        views.append([TF.center_crop(img, (224, 224)) for img in frames])
        
        # Side Crops (Left/Right hoặc Top/Bottom tùy aspect ratio)
        if w > h:
            views.append([TF.crop(img, 0, 0, 224, 224) for img in frames])        # Left
            views.append([TF.crop(img, 0, w-224, 224, 224) for img in frames])    # Right
        else:
            views.append([TF.crop(img, 0, 0, 224, 224) for img in frames])        # Top
            views.append([TF.crop(img, h-224, 0, 224, 224) for img in frames])    # Bottom
        
        # --- 3 FLIPPED CROPS ---
        flipped_views = []
        for v_frames in views:
            flipped_views.append([TF.hflip(img) for img in v_frames])
        
        all_views = views + flipped_views  # Total: 6 views
        
        return torch.stack(view_tensors), video_id  # Shape: (6, T, C, H, W)
```

### 6-View Inference

```python
with torch.no_grad():
    for multi_view_videos, video_ids in test_loader:
        # Shape: (B, 6, T, C, H, W)
        B, V, T, C, H, W = multi_view_videos.shape
        
        # Flatten views into batch: (B*6, T, C, H, W)
        flat_videos = multi_view_videos.view(B * V, T, C, H, W).to(DEVICE)
        
        outputs = model(flat_videos)
        logits = outputs.logits  # (B*6, 51)
        
        # Reshape: (B, 6, 51)
        logits = logits.view(B, V, -1)
        
        # Average over 6 views
        avg_logits = logits.mean(dim=1)  # (B, 51)
        
        preds = avg_logits.argmax(dim=1)
```

### TTA Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    Original Video                            │
│                  (W x H resolution)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Left/Top │ │  Center  │ │Right/Bot │
    │   Crop   │ │   Crop   │ │   Crop   │
    │ 224x224  │ │ 224x224  │ │ 224x224  │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ + Flip   │ │ + Flip   │ │ + Flip   │
    │ (View 4) │ │ (View 5) │ │ (View 6) │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌──────────────┐
              │ Average Pool │
              │   Logits     │
              └──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Prediction  │
              └──────────────┘
```

---

## 📈 Training Techniques Summary

| Technique | Phase 1 | Phase 2 | Purpose |
|-----------|---------|---------|---------|
| **Mixup** | ✅ α=0.8 | ❌ | Regularization, prevent overfitting |
| **Label Smoothing** | ❌ | ✅ 0.1 | Calibration, prevent overconfidence |
| **Learning Rate** | 5e-5 | 1e-6 | Explore → Polish |
| **LR Schedule** | Cosine + Warmup | Cosine | Smooth decay |
| **Gradient Clipping** | ✅ max=1.0 | ✅ max=1.0 | Stability |
| **Mixed Precision** | ✅ FP16 | ✅ FP16 | Memory + Speed |

---

## 💡 Key Insights

### 1. VideoMAE vs Image ViT
- **VideoMAE** được pretrain với Masked Autoencoder trên video
- Học được temporal patterns và motion features
- Kinetics-400 pretrain phù hợp hơn ImageNet cho action recognition

### 2. 2-Stage Training Rationale
- **Phase 1**: Strong augmentation + high LR = diverse feature learning
- **Phase 2**: Clean data + low LR = refinement without forgetting

### 3. Consistent Transform Importance
- Video frames phải giữ spatial coherence
- Random per-frame transforms phá vỡ motion information

### 4. 6-View TTA Benefits
- Robust hơn single-view prediction
- Capture different regions of action
- Horizontal flip handles left/right invariance

---

## 📝 Submission Format

```csv
id,class
0,brush_hair
1,catch
2,clap
...
```

Output file: `submission_multiview_6crops.csv`

---

## 🔧 Dependencies

```python
pip install transformers accelerate evaluate

# Main imports
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from transformers import get_cosine_schedule_with_warmup
```

---

## ⏱️ Training Timeline

| Phase | Duration | Purpose |
|-------|----------|---------|
| Phase 1 | ~6.7 hours | Heavy training với Mixup |
| Phase 2 | ~0.1 hours | Light fine-tuning |
| Inference | ~7 minutes | 6-View TTA |
| **Total** | **~6.8 hours** | |
