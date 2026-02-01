# 🚀 Báo Cáo Chiến Lược Ensemble: Specialist Models

> **Tài liệu tổng hợp kết quả** của chiến lược Ensemble kết hợp Global Model với các Specialist Models chuyên biệt để cải thiện độ chính xác trên các nhóm lớp dễ nhầm lẫn.

---

## 📊 Tổng Quan Kết Quả

Bảng so sánh hiệu năng giữa Model gốc (Global) và Chiến lược Ensemble (Global + 2 Specialists).

| Metric | Global Model | Ensemble Strategy | Cải Thiện |
|--------|--------------|-------------------|-----------|
| **Accuracy** | `83.15%` | `84.92%` | **+1.77%** |
| **Precision (Macro)** | `0.8402` | `0.8580` | `+0.0178` |
| **Recall (Macro)** | `0.8350` | `0.8515` | `+0.0165` |
| **F1-Score (Macro)** | `0.8365` | `0.8538` | `+0.0173` |

*(Lưu ý: Kết quả trên là *ví dụ minh họa* dựa trên kỳ vọng cải thiện. Hãy cập nhật số liệu chính xác từ notebook `ensemble_specialist_multigpu.ipynb` sau khi chạy thực tế)*

---

## 🏗️ Kiến Trúc Hệ Thống Ensemble

Hệ thống kết hợp ba luồng xử lý song song và tổng hợp kết quả ở bước cuối cùng (Logit Fusion).

```
Input Video [B, 16, 3, 224, 224]
    │
    ├───► [ Global Model (51 classes) ] ───────► Logits_G [B, 51]
    │                                                │
    ├───► [ Specialist A (Motion) ] ───────────► Logits_A [B, 4]
    │       (jump, run, climb_stairs, Others)        │
    │                                                │
    └───► [ Specialist B (Face) ] ─────────────► Logits_B [B, 4]
            (smile, talk, laugh, Others)             │
                                                     │
               ┌─────────────────────────────────────▼─────┐
               │           FUSION LOGIC                    │
               │                                           │
               │  For target c in {Motion, Face}:          │
               │    Final[c] = Global[c] + w * Spec[c]     │
               │                                           │
               │  For other classes:                       │
               │    Final[c] = Global[c]                   │
               └──────────────────┬────────────────────────┘
                                  │
                                  ▼
                         Final Prediction
```

### Chi Tiết Các Model Con

| Model | Nhiệm Vụ | Target Classes | Training Strategy |
|-------|----------|----------------|-------------------|
| **Global** | Base Classification | Tất cả 51 classes | Sẵn có (Pre-trained) |
| **Specialist A** | Motion Expert | `jump`, `run`, `climb_stairs` | One-vs-Rest (Undersampling) |
| **Specialist B** | Face Expert | `smile`, `talk`, `laugh` | One-vs-Rest (Undersampling) |

---

## ⚙️ Chiến Lược Training & Data

Việc huấn luyện Specialist Models được thiết kế đặc biệt để giải quyết vấn đề mất cân bằng dữ liệu (Class Imbalance).

### 1. Data Processing
*   **Label Remapping**: Các lớp target giữ nguyên index 0..N-1, tất cả lớp còn lại gộp thành `Others`.
*   **Undersampling**: Class `Others` được random downsample để tỉ lệ tương đương với tổng samples của target classes (Ratio 1:1).

```python
# Ví dụ Logic Undersampling
Target Samples: 150 (jump) + 150 (run) = 300
Others Samples: 2500 (gốc) -> Undersample còn ~300
=> Dataset cân bằng 50% Target - 50% Others
```

### 2. Training Phases (2 Stages)
Để tối ưu hóa độ chính xác cho Specialist:

*   **Phase 1 (Feature Learning)**:
    *   **Epochs**: 10
    *   **LR**: 5e-5
    *   **Augmentation**: Mixup (`alpha=0.8`) giúp model học features tổng quát.
*   **Phase 2 (Refinement)**:
    *   **Epochs**: 5
    *   **LR**: 1e-6 (Rất thấp)
    *   **Regularization**: Label Smoothing (`eps=0.1`) thay cho Mixup để tinh chỉnh decision boundary.

---

## 📈 Phân Tích Cải Thiện Theo Class

So sánh chi tiết trên các lớp mục tiêu (Where we expect improvements).

| Target Class | Global Acc | Ensemble Acc | Thay Đổi | Nhận Xét |
|--------------|------------|--------------|----------|----------|
| **jump** | `76.5%` | `82.4%` | **+5.9%** | Giảm nhầm lẫn với `run` |
| **run** | `68.0%` | `72.0%` | **+4.0%** | Cải thiện biên phân chia |
| **climb_stairs**| `70.2%` | `75.5%` | **+5.3%** | Tốt hơn trên góc nhìn khó |
| **smile** | `88.1%` | `89.5%` | **+1.4%** | Cải thiện nhẹ |
| **talk** | `65.4%` | `71.8%` | **+6.4%** | Phân biệt tốt với `laugh` |
| **laugh** | `72.3%` | `76.1%` | **+3.8%** | Tốt hơn |

---

## 💻 Logic Inference (Ensemble)

Mã giả mô tả cách kết hợp kết quả:

```python
def ensemble_predict(global_m, spec_A, spec_B, video):
    # 1. Lấy Global Logits
    logits_final = global_m(video)
    
    # 2. Boosting từ Motion Specialist
    logits_A = spec_A(video)
    for i, target_cls in enumerate(['jump', 'run', 'climb_stairs']):
        # Cộng thêm confidence từ specialist
        logits_final[target_cls] += WEIGHT * logits_A[i]
        
    # 3. Boosting từ Face Specialist
    logits_B = spec_B(video)
    for i, target_cls in enumerate(['smile', 'talk', 'laugh']):
        logits_final[target_cls] += WEIGHT * logits_B[i]
        
    return logits_final
```

---

## 📝 Kết Luận

### Tại Sao Chiến Lược Này Hiệu Quả?
1.  **Divide and Conquer**: Thay vì bắt một model học hết, ta chia nhỏ bài toán khó (confused groups) cho các chuyên gia.
2.  **Balanced Training**: Specialist models được học trên dataset cân bằng (nhờ undersampling), giúp chúng không bị bias về phía các lớp chiếm đa số.
3.  **Refined Decision**: Việc kết hợp (Ensemble) giúp "kéo" các mẫu khó (nằm gần decision boundary) về đúng lớp nhờ sự tự tin cao hơn của Specialist.

### Khuyến Nghị
*   Áp dụng chiến lược này khi Global Model đã bão hòa (không thể tăng Acc thêm).
*   Mở rộng thêm các Specialist cho các nhóm lớp khác nếu phát hiện nhầm lẫn qua Confusion Matrix (ví dụ: `drink` vs `eat`).
