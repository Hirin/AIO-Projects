# 🚀 Báo Cáo Chiến Lược Ensemble Cải Tiến: TTA + Balanced Sampling + Adaptive Gating

> **Tài liệu tổng hợp kết quả** của phiên bản cải tiến chiến lược Ensemble, tích hợp ba kỹ thuật nâng cao: Test-Time Augmentation (TTA), Class-Balanced Sampling, và Adaptive Confidence-based Gating.

---

## 📊 Tổng Quan Kết Quả

Bảng so sánh hiệu năng giữa Model gốc (Global) và Chiến lược Ensemble Cải Tiến.

| Metric | Global Model | Ensemble Improved | Cải Thiện |
|--------|--------------|-------------------|-----------|
| **Accuracy** | `85.10%` | `86.00%` | **+0.90%** |
| **Precision (Macro)** | `0.8535` | `0.8603` | `+0.0068` |
| **Recall (Macro)** | `0.8545` | `0.8608` | `+0.0063` |
| **F1-Score (Macro)** | `0.8530` | `0.8600` | `+0.0070` |
| **F1-Score (Weighted)** | `0.8528` | `0.8596` | `+0.0068` |

> ✅ **Specialist Usage**: 30.6% mẫu test sử dụng specialist models (chỉ khi cần thiết nhờ Adaptive Gating)

---

## 🆕 Ba Cải Tiến Chính

### 1️⃣ Test-Time Augmentation (TTA) - Multi-view Testing

Thay vì sử dụng single CenterCrop như phiên bản gốc, áp dụng **6 views per video**:

| View | Temporal Crop | Spatial Crop |
|------|---------------|--------------|
| 1 | Start (0-70%) | Left |
| 2 | Start (0-70%) | Center |
| 3 | Start (0-70%) | Right |
| 4 | End (30-100%) | Left |
| 5 | End (30-100%) | Center |
| 6 | End (30-100%) | Right |

**Logic Inference:**
```python
# Average logits across 6 views
final_logits = mean([model(view) for view in 6_views])
```

**Config:**
```python
USE_TTA = True  # Flag bật/tắt TTA
```

---

### 2️⃣ Class-Balanced Sampling (Thay Undersampling)

**Vấn đề với Undersampling gốc:**
- Mất dữ liệu quý giá từ class "Others"
- Không tận dụng được toàn bộ dataset

**Giải pháp mới - WeightedRandomSampler:**

```python
# Giữ TOÀN BỘ dữ liệu
class_counts = [465, 5789]  # targets, others
weights = 1.0 / class_counts  # Inverse frequency
sampler = WeightedRandomSampler(sample_weights, len(samples))
```

| So sánh | Undersampling (cũ) | Balanced Sampling (mới) |
|---------|-------------------|------------------------|
| Samples | ~930 (sau undersample) | 6254 (toàn bộ) |
| Others | Bị cắt bớt | Giữ nguyên |
| Training | shuffle=True | sampler + shuffle=False |

---

### 3️⃣ Adaptive Ensemble - Confidence-based Gating

**Vấn đề với Fixed-weight Fusion gốc:**
- Luôn gọi specialist cho MỌI sample
- Tốn compute không cần thiết
- Có thể gây nhiễu với các mẫu global đã confident cao

**Giải pháp mới - Conditional Specialist:**

```python
CONFIDENCE_THRESHOLD = 0.7
SENSITIVE_CLASSES = ['jump', 'run', 'climb_stairs', 'smile', 'talk', 'laugh']

# Chỉ gọi specialist khi:
if confidence < THRESHOLD or pred_class in SENSITIVE_CLASSES:
    final_logits = global_logits + specialist_logits
else:
    final_logits = global_logits  # Không cần specialist
```

**Lợi ích:**
- Giảm 69.4% compute (chỉ 30.6% samples cần specialist)
- Tránh nhiễu cho các mẫu đã confident cao
- Tập trung resource vào mẫu khó

---

## 🏗️ Kiến Trúc Hệ Thống Cải Tiến

```
Input Video [B, 16, 3, 224, 224]
    │
    ├──► [ TTA: Generate 6 Views ]
    │           │
    │           ├──► View 1 (Start, Left)
    │           ├──► View 2 (Start, Center)
    │           ├──► ...
    │           └──► View 6 (End, Right)
    │
    ├──► [ Global Model (51 classes) ] ─► Average Logits_G [B, 51]
    │           │
    │           ▼
    │    ┌─────────────────────────────────┐
    │    │    CONFIDENCE CHECK             │
    │    │    conf = max(softmax(Logits_G))│
    │    │    pred = argmax(Logits_G)      │
    │    └──────────────┬──────────────────┘
    │                   │
    │         ┌─────────▼─────────┐
    │         │ conf < 0.7 OR     │
    │         │ pred in SENSITIVE?│
    │         └─────────┬─────────┘
    │                   │
    │         [YES]     │     [NO]
    │           │       │       │
    │           ▼       │       │
    │    ┌──────────┐   │       │
    ├──► │Spec A    │   │       │
    │    │(Motion)  │───┤       │
    │    └──────────┘   │       │
    │    ┌──────────┐   │       │
    └──► │Spec B    │   │       │
         │(Face)    │───┤       │
         └──────────┘   │       │
                        ▼       ▼
              FUSION LOGITS   GLOBAL ONLY
                        │       │
                        └───┬───┘
                            ▼
                    Final Prediction
```

---

## ⚙️ Training Configuration

### Specialist Models (Motion & Face)

| Parameter | Phase 1 (Mixup) | Phase 2 (Label Smoothing) |
|-----------|-----------------|--------------------------|
| Epochs | 10 | 5 |
| Learning Rate | 5e-5 | 1e-6 |
| Augmentation | Mixup (α=0.8) | Label Smoothing (ε=0.1) |
| Sampler | WeightedRandomSampler | WeightedRandomSampler |
| Batch Size | 16 (8 × 2 GPUs) | 16 (8 × 2 GPUs) |

### Training Results

| Specialist | Best Train Acc | Phase |
|------------|---------------|-------|
| **Motion** (jump, run, climb_stairs) | 99.98% | Phase 2 |
| **Face** (smile, talk, laugh) | 100.00% | Phase 2 |

---

## 📈 Phân Tích Cải Thiện Theo Class

So sánh chi tiết trên các lớp mục tiêu:

| Target Class | Global Acc | Ensemble Acc | Thay Đổi |
|--------------|------------|--------------|----------|
| **jump** | 69.0% | 72.4% | **+3.4%** |
| **run** | 60.0% | 62.9% | **+2.9%** |
| **climb_stairs** | 58.8% | 58.8% | **+0.0%** |
| **smile** | 66.7% | 66.7% | **+0.0%** |
| **talk** | 88.2% | 88.2% | **+0.0%** |
| **laugh** | 57.1% | 57.1% | **+0.0%** |

> *Lưu ý: Các class smile, talk, laugh không thay đổi do confidence cao hoặc không thuộc nhóm cần specialist*

---

## 💻 Code Highlights

### TTATestDataset Class

```python
class TTATestDataset(Dataset):
    """
    Test dataset with TTA support: 3 spatial crops × 2 temporal crops = 6 views.
    """
    def _get_spatial_crop(self, img, crop_type):
        # Apply left, center, or right crop
        ...
    
    def _load_frames_temporal(self, vid_dir, temporal_position):
        # Load start (0-70%) or end (30-100%) segment
        ...
    
    def __getitem__(self, idx):
        if USE_TTA:
            # Generate 6 views
            return torch.stack(views), vid_id  # [6, T, C, H, W]
        else:
            return single_view, vid_id
```

### Adaptive Ensemble Predict

```python
@torch.no_grad()
def adaptive_ensemble_predict(global_m, spec_A, spec_B, video, debug=False):
    # Get global predictions
    logits_global = global_m(video)
    probs = F.softmax(logits_global, dim=1)
    confidence, pred_class = probs.max(dim=1)
    
    # Determine which samples need specialist
    needs_specialist = (confidence < THRESHOLD) | (pred_class in SENSITIVE)
    
    if needs_specialist.any():
        # Only run specialists when needed
        logits_a = spec_A(video)
        logits_b = spec_B(video)
        # Apply fusion only for needed samples
        ...
    
    return final_logits
```

---

## 📝 Kết Luận

### Tại Sao Các Cải Tiến Này Hiệu Quả?

| Kỹ thuật | Lợi ích |
|----------|---------|
| **TTA Multi-view** | Robust predictions qua averaging nhiều views |
| **Balanced Sampling** | Không mất data, training ổn định hơn |
| **Adaptive Gating** | Tiết kiệm compute, tránh nhiễu false positive |

### So Sánh với Phiên Bản Gốc

| Aspect | Original | Improved |
|--------|----------|----------|
| Test-time views | 1 (Center) | 6 (Multi-view) |
| Data strategy | Undersampling | WeightedRandomSampler |
| Specialist usage | Always (100%) | Conditional (30.6%) |
| Accuracy | 83.15% | 86.00% |

### File Notebook

📓 [ensemble_specialist_improved.ipynb](../notebooks/ensemble_specialist_improved.ipynb)

---

*Báo cáo được tạo tự động từ kết quả chạy notebook ngày 2026-02-01*
