# Baseline BTC - Video Action Recognition Analysis

> **Tài liệu phân tích chi tiết** về notebook `baseline-btc.ipynb` - Giải pháp baseline cho bài toán Video Action Recognition trên dataset HMDB51.

---

## � Tổng Quan

| Thông tin | Chi tiết |
|-----------|----------|
| **Bài toán** | Video Action Recognition (Nhận dạng hành động trong video) |
| **Dataset** | HMDB51 (51 classes) |
| **Backbone** | ViT-Small (ImageNet pretrained) |
| **Framework** | PyTorch + timm |
| **Platform** | Kaggle (Tesla T4 GPU) |
| **Training Time** | ~21 phút (4 epochs) |

---

## 🏗️ Kiến Trúc Model

### LightweightViTForAction

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO                               │
│                  [B, T, C, H, W]                             │
│            (Batch, 16 Frames, 3, 224, 224)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              RESHAPE TO FRAMES                               │
│                 [B*T, C, H, W]                               │
│              (B*16, 3, 224, 224)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               ViT-Small Backbone                             │
│        (vit_small_patch16_224 - ImageNet pretrained)         │
│                                                              │
│    • Patch Size: 16x16                                       │
│    • Hidden Dim: 384                                         │
│    • Heads: 6                                                │
│    • Layers: 12                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FRAME FEATURES                                  │
│                [B*T, embed_dim]                              │
│                  (B*16, 384)                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              RESHAPE BACK                                    │
│              [B, T, embed_dim]                               │
│                (B, 16, 384)                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           TEMPORAL MEAN POOLING                              │
│              [B, embed_dim]                                  │
│                 (B, 384)                                     │
│                                                              │
│    pooled = features.mean(dim=1)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            LINEAR CLASSIFICATION HEAD                        │
│              nn.Linear(384, 51)                              │
│                                                              │
│                 [B, num_classes]                             │
│                   (B, 51)                                    │
└─────────────────────────────────────────────────────────────┘
```

### Đặc điểm Model

| Component | Specification |
|-----------|---------------|
| **Backbone** | `vit_small_patch16_224` |
| **Parameters** | ~22M |
| **Model Size** | ~84 MB |
| **Embed Dim** | 384 |
| **Temporal Modeling** | Mean Pooling (không có learnable temporal layer) |
| **Classification Head** | Single Linear layer |

---

## ⚙️ Configuration & Hyperparameters

### Data Parameters

```python
PATH_DATA_TRAIN = '/kaggle/input/action-video/data/data_train'
PATH_DATA_TEST = '/kaggle/input/action-video/data/test'

NUM_FRAMES = 16        # Số frames lấy từ mỗi video
FRAME_STRIDE = 2       # Bước nhảy khi sampling frames
IMG_SIZE = 224         # Kích thước ảnh đầu vào
```

### Training Parameters

```python
BATCH_SIZE = 16
EPOCHS = 4
BASE_LR = 1e-4         # Learning rate cho backbone
HEAD_LR = 5e-4         # Learning rate cho classification head (5x cao hơn)
WEIGHT_DECAY = 0.05
GRAD_ACCUM_STEPS = 4   # Effective batch size = 16 * 4 = 64
```

---

## � Data Pipeline

### 1. Frame Sampling Strategy

```python
def _select_indices(self, total):
    """
    Uniform sampling với stride từ video.
    
    Ví dụ: video 100 frames, NUM_FRAMES=16, FRAME_STRIDE=2
    -> steps = max(16*2, 16) = 32
    -> grid = linspace(0, 99, 32) = [0, 3.2, 6.4, ..., 99]
    -> idxs = grid[::2] = lấy mỗi 2 bước = 16 frames
    """
    steps = max(self.num_frames * self.frame_stride, self.num_frames)
    grid = torch.linspace(0, total - 1, steps=steps)
    idxs = grid[::self.frame_stride].long()
    
    # Padding nếu không đủ frames
    if idxs.numel() < self.num_frames:
        pad = idxs.new_full((self.num_frames - idxs.numel(),), idxs[-1].item())
        idxs = torch.cat([idxs, pad], dim=0)
    
    return idxs[:self.num_frames]
```

### 2. Data Augmentation (VideoTransform)

#### Training Augmentation

```python
# 1. Random Scale (0.8 - 1.0)
scale = random.uniform(0.8, 1.0)
new_h, new_w = int(h * scale), int(w * scale)

# 2. Random Crop to 224x224
i = random.randint(0, max(0, new_h - 224))
j = random.randint(0, max(0, new_w - 224))

# 3. Resize nếu cần
frames = TF.resize(frames, [224, 224])

# 4. Random Horizontal Flip (p=0.5)
if random.random() < 0.5:
    frames = TF.hflip(frames)

# 5. Normalize
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
```

#### Test Augmentation

```python
# Chỉ Resize + Normalize (không random transform)
frames = TF.resize(frames, [224, 224])
normalized = TF.normalize(frame, mean, std)
```

> ⚠️ **Lưu ý**: Baseline không sử dụng consistent spatial transform - mỗi frame có thể bị crop/flip khác nhau trong cùng một video.

---

## 🏋️ Training Loop

### Mixed Precision Training với Gradient Accumulation

```python
def train_one_epoch(model, loader, optimizer, scaler, device, grad_accum_steps=1):
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (videos, labels) in enumerate(loader):
        # Forward với AMP
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)
        
        # Scale loss cho gradient accumulation
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        
        # Update weights mỗi grad_accum_steps batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### Differential Learning Rates

```python
# Backbone (ViT): Lower LR để giữ pretrained features
backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]

# Head (Linear): Higher LR để học nhanh task mới
head_params = [p for n, p in model.named_parameters() if 'head' in n]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},   # BASE_LR
    {"params": head_params, "lr": 5e-4},       # HEAD_LR (5x)
], weight_decay=0.05)
```

---

## 🔍 Inference

### Standard Single-View Inference

```python
model.eval()
with torch.no_grad():
    for videos, video_ids in test_loader:
        videos = videos.to(DEVICE)
        logits = model(videos)
        preds = logits.argmax(dim=1)  # Hard prediction
```

> ⚠️ **Không có TTA (Test Time Augmentation)** - Chỉ sử dụng center resize, không flip hay multi-crop.

---

## 📁 Dataset Structure

### Training Data

```
data_train/
├── brush_hair/
│   ├── video_001/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   ├── video_002/
│   └── ...
├── cartwheel/
├── catch/
└── ... (51 classes)
```

### Test Data

```
test/
├── 0/
│   ├── frame_0001.jpg
│   └── ...
├── 1/
├── 2/
└── ... (video IDs)
```

---

## � Kết Quả Training

| Epoch | Train Loss | Train Acc |
|-------|-----------|-----------|
| 1 | - | ~0.4x |
| 2 | - | ~0.5x |
| 3 | - | ~0.6x |
| 4 | - | ~0.65-0.70 |

---

## ⚠️ Hạn Chế của Baseline

| Vấn đề | Mô tả |
|--------|-------|
| **No Temporal Learning** | Mean pooling không capture được motion patterns giữa các frames |
| **Weak Augmentation** | Không có Mixup/CutMix, dễ bị overfit |
| **Short Training** | Chỉ 4 epochs, model chưa converge hoàn toàn |
| **No TTA** | Single view inference kém robust |
| **Inconsistent Transforms** | Mỗi frame có thể bị transform khác nhau |
| **ImageNet Pretrain Only** | Pretrain trên ảnh tĩnh, không phù hợp cho video understanding |

---

## � Code Snippets Quan Trọng

### Model Definition

```python
class LightweightViTForAction(nn.Module):
    def __init__(self, num_classes=51, pretrained_name='vit_small_patch16_224'):
        super().__init__()
        self.vit = timm.create_model(pretrained_name, pretrained=True, num_classes=0)
        self.embed_dim = self.vit.num_features  # 384
        self.head = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, video):
        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)
        features = self.vit(x)  # [B*T, 384]
        features = features.view(B, T, self.embed_dim)
        pooled = features.mean(dim=1)  # Temporal pooling
        return self.head(pooled)
```

### Checkpoint Saving

```python
torch.save({
    'model': model.state_dict(),
    'classes': train_dataset.classes,
    'acc': best_acc
}, checkpoint_path)
```

---

## 📝 Submission Format

```csv
id,class
0,brush_hair
1,catch
2,clap
...
```

---

## � Hướng Cải Tiến Tiềm Năng

1. **Thay backbone**: Sử dụng video-pretrained models
2. **Thêm temporal modeling**: LSTM, Transformer layers sau ViT
3. **Augmentation mạnh hơn**: Mixup, CutMix, RandAugment
4. **Regularization**: Label Smoothing, Dropout, DropPath
5. **Training dài hơn**: 20-40 epochs với LR scheduler
6. **TTA**: Multi-crop, Multi-scale, Flip ensemble
