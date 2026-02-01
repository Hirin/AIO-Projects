# 📋 BÁO CÁO TỔNG HỢP DỰ ÁN: Video Action Recognition với Chiến Lược Ensemble

> **Dự án:** AIO-Projects / Project-7.1  
> **Nhiệm vụ:** Phân loại hành động (Action Classification) từ video  
> **Mô hình:** VideoMAE (Video Masked AutoEncoder)  
> **Ngày báo cáo:** 2026-02-01

---

## 📑 Mục Lục

1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Phương Pháp & Chiến Lược](#-phương-pháp--chiến-lược)
3. [Kết Quả Thực Nghiệm](#-kết-quả-thực-nghiệm)
4. [Phân Tích Chi Tiết](#-phân-tích-chi-tiết)
5. [Những Cải Thiện Đạt Được](#-những-cải-thiện-đạt-được)
6. [Hạn Chế Còn Tồn Tại](#-hạn-chế-còn-tồn-tại)
7. [Đề Xuất Hướng Phát Triển](#-đề-xuất-hướng-phát-triển)
8. [Kết Luận](#-kết-luận)

---

## 🎯 Tổng Quan Dự Án

### Bài Toán

Phân loại video thành **51 lớp hành động** (action classes) dựa trên dữ liệu video được trích xuất thành các frame ảnh.

### Thách Thức Chính

| Thách Thức | Mô Tả |
|------------|-------|
| **Imbalanced Data** | Phân bố không đều giữa các lớp |
| **Confused Classes** | Một số lớp có đặc trưng tương tự, dễ nhầm lẫn |
| **Computational Cost** | VideoMAE yêu cầu tài nguyên GPU đáng kể |

### Các Lớp Thường Bị Nhầm Lẫn

| Nhóm | Classes | Lý Do Nhầm Lẫn |
|------|---------|----------------|
| **Motion** | `jump`, `run`, `climb_stairs` | Chuyển động cơ thể tương tự |
| **Face** | `smile`, `talk`, `laugh` | Biểu cảm khuôn mặt tương tự |

---

## 🧩 Phương Pháp & Chiến Lược

### Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                              │
│              [Batch, 16 frames, 3, 224, 224]                │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Global   │   │Specialist│   │Specialist│
    │ Model    │   │    A     │   │    B     │
    │(51 class)│   │ (Motion) │   │ (Face)   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         ▼              ▼              ▼
    [Logits_G]    [Logits_A]    [Logits_B]
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  ENSEMBLE LOGIC │
              │  (Fusion/Gating)│
              └────────┬────────┘
                       │
                       ▼
              [ Final Prediction ]
```

### Hai Phiên Bản Chiến Lược

| Phiên Bản | Đặc Điểm Chính |
|-----------|----------------|
| **V1: Ensemble Original** | Fixed-weight fusion, Undersampling, Single-view |
| **V2: Ensemble Improved** | Adaptive Gating, Balanced Sampling, TTA Multi-view |

---

## 📊 Kết Quả Thực Nghiệm

### Bảng So Sánh Tổng Hợp

| Metric | Global Only | Ensemble V1 | Ensemble V2 (Improved) |
|--------|-------------|-------------|------------------------|
| **Accuracy** | 83.15% | 83.15% | **86.00%** |
| **Precision (Macro)** | 0.8343 | 0.8326 | **0.8603** |
| **Recall (Macro)** | 0.8354 | 0.8337 | **0.8608** |
| **F1-Score (Macro)** | 0.8324 | 0.8308 | **0.8600** |
| **F1-Score (Weighted)** | 0.8335 | 0.8318 | **0.8596** |

### Tiến Triển Accuracy

```
Global Only    ████████████████████████████████░░░░░░  83.15%
Ensemble V1    ████████████████████████████████░░░░░░  83.15%
Ensemble V2    █████████████████████████████████████░  86.00%
                                                        ▲
                                                   +2.85%
```

### Kết Quả Training Specialist Models

| Specialist | Target Classes | Train Acc (V1) | Train Acc (V2) |
|------------|----------------|----------------|----------------|
| **Motion** | jump, run, climb_stairs + Others | 96.88% | 99.98% |
| **Face** | smile, talk, laugh + Others | 96.88% | 100.00% |

---

## 🔍 Phân Tích Chi Tiết

### Hiệu Năng Trên Các Lớp Mục Tiêu

| Class | Global | Ensemble V1 | Ensemble V2 | Δ (V1→V2) |
|-------|--------|-------------|-------------|-----------|
| **jump** | 69.0% | 69.0% | 72.4% | **+3.4%** |
| **run** | 60.0% | 60.0% | 62.9% | **+2.9%** |
| **climb_stairs** | 58.8% | 58.8% | 58.8% | +0.0% |
| **smile** | 66.7% | 66.7% | 66.7% | +0.0% |
| **talk** | 88.2% | - | 88.2% | +0.0% |
| **laugh** | 57.1% | - | 57.1% | +0.0% |

### Tại Sao Ensemble V1 Không Cải Thiện?

> [!WARNING]
> **Vấn đề của Ensemble V1:**
> - Fixed-weight fusion (0.4) áp dụng cho MỌI sample
> - Undersampling làm mất dữ liệu (~930 samples thay vì 6254)
> - Single-view testing không robust

### Tại Sao Ensemble V2 Hiệu Quả Hơn?

| Kỹ Thuật | Lợi Ích |
|----------|---------|
| **TTA Multi-view** | 6 views/video → robust averaging |
| **Balanced Sampling** | Giữ 100% data (6254 samples) |
| **Adaptive Gating** | Chỉ gọi specialist khi cần (30.6%) |

---

## ✅ Những Cải Thiện Đạt Được

### 1. Accuracy Tổng Thể

```diff
- Ensemble V1: 83.15% (không cải thiện so với Global)
+ Ensemble V2: 86.00% (+2.85% so với Global)
```

### 2. Hiệu Quả Sử Dụng Tài Nguyên

| Aspect | V1 | V2 |
|--------|----|----|
| Specialist calls | 100% samples | 30.6% samples |
| Compute saved | 0% | **69.4%** |

### 3. Sử Dụng Dữ Liệu

| Aspect | V1 (Undersampling) | V2 (Balanced) |
|--------|-------------------|---------------|
| Training samples | ~930 | 6254 |
| Data utilization | ~15% | **100%** |

### 4. Robust Predictions

| Aspect | V1 | V2 |
|--------|----|----|
| Test-time views | 1 (Center only) | 6 (Multi-view TTA) |
| Variance reduction | None | Averaging over views |

---

## ⚠️ Hạn Chế Còn Tồn Tại

### 1. Hiệu Năng Trên Một Số Classes Vẫn Thấp

| Class | Accuracy | Vấn Đề |
|-------|----------|--------|
| `climb_stairs` | 58.8% | Không cải thiện dù có specialist |
| `laugh` | 57.1% | Confusion với `smile`, `talk` |
| `run` | 62.9% | Vẫn thấp dù đã cải thiện |

> [!CAUTION]
> **Root Cause:** Các lớp này có đặc trưng visual rất tương tự, khó phân biệt chỉ từ video frames.

### 2. Specialist Models Có Thể Gây Nhiễu

- Khi Global Model đã confident cao, việc thêm specialist logits có thể:
  - Làm "dilute" confidence
  - Đổi prediction sang class sai

**Giải pháp hiện tại:** Adaptive Gating chỉ áp dụng specialist khi confidence < 0.7

### 3. Computational Cost Vẫn Cao

| Processing | Time (estimated) |
|------------|------------------|
| 1 video (no TTA) | ~0.5s |
| 1 video (6-view TTA) | ~3s |
| 1000 videos (TTA + Ensemble) | ~50 phút |

### 4. Hyper-parameters Cần Tinh Chỉnh

```python
CONFIDENCE_THRESHOLD = 0.7  # Cần tuning
FUSION_WEIGHT = 0.4          # Cần tuning
SENSITIVE_CLASSES = [...]    # Cần validate
```

Các giá trị này được chọn theo kinh nghiệm, chưa có grid search hoặc optimization.

### 5. Class Imbalance Trong Test Set

Một số classes có rất ít samples trong test set, khiến metrics không ổn định:

| Ít Samples | Nhiều Samples |
|------------|---------------|
| laugh (7) | talk (17) |
| climb_stairs (17) | jump (29) |

---

## 💡 Đề Xuất Hướng Phát Triển

### Ngắn Hạn (Quick Wins)

| Đề Xuất | Khó Độ | Impact |
|---------|--------|--------|
| Tune `CONFIDENCE_THRESHOLD` | ⭐ | Cao |
| Tune `FUSION_WEIGHT` | ⭐ | Trung bình |
| Thêm TTA views (5-crop) | ⭐⭐ | Trung bình |

### Trung Hạn

| Đề Xuất | Khó Độ | Impact |
|---------|--------|--------|
| **Temporal Modeling** | ⭐⭐⭐ | Cao |
| Sử dụng optical flow | ⭐⭐⭐ | Cao |
| Hierarchical Classification | ⭐⭐ | Trung bình |

### Dài Hạn

| Đề Xuất | Khó Độ | Impact |
|---------|--------|--------|
| Audio-Visual Fusion | ⭐⭐⭐⭐ | Rất cao |
| Self-training / Semi-supervised | ⭐⭐⭐ | Cao |
| Model Distillation | ⭐⭐⭐ | Trung bình |

---

## 🎯 Kết Luận

### Thành Công

✅ Đạt **86.00% accuracy** (+2.85% so với Global baseline)  
✅ Giảm **69.4% compute** nhờ Adaptive Gating  
✅ Sử dụng **100% training data** nhờ Balanced Sampling  
✅ Robust predictions nhờ **TTA Multi-view**  

### Bài Học Rút Ra

| Insight | Chi Tiết |
|---------|----------|
| **Undersampling không tối ưu** | Mất dữ liệu → mất thông tin |
| **Fixed fusion gây nhiễu** | Cần conditional/adaptive approach |
| **TTA hiệu quả** | Multi-view averaging giảm variance |
| **Specialist cần careful design** | Không phải lúc nào cũng giúp ích |

### Kết Quả Cuối Cùng

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    📈 BEST MODEL: Ensemble V2 (Improved)                    │
│                                                             │
│    Accuracy:        86.00%                                  │
│    F1-Score:        0.8600 (Macro)                          │
│    Specialist Use:  30.6% (Adaptive)                        │
│                                                             │
│    Notebooks:                                               │
│    - ensemble_specialist_multigpu.ipynb (V1)                │
│    - ensemble_specialist_improved.ipynb (V2)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Tài Liệu Tham Khảo

| File | Mô Tả |
|------|-------|
| [ensemble_results_summary.md](./ensemble_results_summary.md) | Báo cáo chi tiết Ensemble V1 |
| [ensemble_improved_results_summary.md](./ensemble_improved_results_summary.md) | Báo cáo chi tiết Ensemble V2 |
| [ensemble_specialist_multigpu.ipynb](../notebooks/ensemble_specialist_multigpu.ipynb) | Notebook V1 |
| [ensemble_specialist_improved.ipynb](../notebooks/ensemble_specialist_improved.ipynb) | Notebook V2 |

---

*Báo cáo tổng hợp - Cập nhật ngày 2026-02-01*
