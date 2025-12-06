# 📈 Dự Đoán Giá Cổ Phiếu - LTSF-Linear với HMM Regime-Switching

![Project Banner](https://img.shields.io/badge/Project-Time%20Series%20Forecasting-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

## 📖 Tổng Quan

Dự án dự đoán giá đóng cửa cổ phiếu (**FPT**, **VIC**) trong **100 ngày tiếp theo** sử dụng **LTSF-Linear** kết hợp **HMM Regime-Switching**.

### Ý tưởng chính

```
┌─────────────────────────────────────────────────────────────────┐
│  HMM REGIME-SWITCHING APPROACH                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. HMM phát hiện "tâm lý thị trường" ẩn (regimes)              │
│     - Regime 0: Stable (volatility thấp)                        │
│     - Regime 1: Transition                                      │
│     - Regime 2: Volatile (volatility cao)                       │
│                                                                 │
│  2. Train model RIÊNG cho từng regime                           │
│     → Model học pattern của từng điều kiện thị trường           │
│                                                                 │
│  3. Predict dựa trên current regime (regime ngày cuối)          │
│     → Dùng model phù hợp với điều kiện hiện tại                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Tính Năng Chính

| Feature | Mô tả |
|---------|-------|
| **Models** | Linear, DLinear (Trend/Seasonal decomposition) |
| **Variants** | Univariate (close only) / Multivariate (close, volume, spreads) |
| **RevIN** | Reversible Instance Normalization - xử lý distribution shift |
| **HMM** | Hidden Markov Model - phân loại regime thị trường |
| **Grid Search** | Tự động thử hàng trăm tổ hợp tham số |

## 🔄 Pipeline Chi Tiết

### Tổng quan Flow

```mermaid
flowchart TB
    subgraph STEP1["STEP 1: DATA LOADING"]
        A[("🗃️ Raw Data<br/>1149 x 6")]
    end
    
    subgraph STEP2["STEP 2: DATA SPLITTING"]
        direction LR
        B1["🔵 TRAIN<br/>839 days"]
        B2["🟡 VAL<br/>210 days"]
        B3["🔴 TEST<br/>100 days"]
    end
    
    subgraph STEP3["STEP 3: FEATURE ENGINEERING"]
        direction LR
        C1["📊 Log Transform"]
        C2["📈 Spread Features"]
        C3["📉 HMM Features"]
    end
    
    subgraph STEP4["STEP 4: HMM REGIME DETECTION"]
        D1["fit on TRAIN"]
        D2["predict on TRAIN+VAL"]
        D3["Regime Labels: 0, 1, 2"]
    end
    
    subgraph STEP5["STEP 5: MODEL TRAINING"]
        direction LR
        E1["Model 0<br/>Stable"]
        E2["Model 1<br/>Transition"]
        E3["Model 2<br/>Volatile"]
    end
    
    subgraph STEP6["STEP 6: EVALUATION"]
        F1["Grid Search on VAL"]
        F2["Final Eval on TEST"]
    end
    
    subgraph STEP7["STEP 7: PRODUCTION"]
        G1["Retrain on 95% full data"]
        G2["Select model by regimes[-1]"]
    end
    
    subgraph OUTPUT["FINAL: SUBMISSION"]
        H[("📄 submission.csv<br/>100 days forecast")]
    end
    
    A --> B1 & B2 & B3
    B1 & B2 --> C1 & C2 & C3
    B3 -.->|"for comparison"| F2
    C1 & C2 & C3 --> D1
    D1 --> D2 --> D3
    D3 --> E1 & E2 & E3
    E1 & E2 & E3 --> F1
    F1 --> F2
    F2 --> G1 --> G2 --> H
```

### Step 1: Feature Engineering

```mermaid
flowchart LR
    A["Raw Data<br/>OHLCV"] --> B["Log Transform"]
    A --> C["Spread Features"]
    B --> D["close_log<br/>volume_log"]
    C --> E["HL_Spread<br/>OC_Spread"]
    D & E --> F["HMM Features"]
    F --> G["returns<br/>volatility<br/>trend"]
```

### Step 2: Data Splitting

```mermaid
pie title Data Split (1149 days)
    "TRAIN (839 - 73%)" : 839
    "VAL (210 - 18%)" : 210
    "TEST (100 - 9%)" : 100
```

> **Phân chia dữ liệu:**
> - **TRAIN**: Để train model
> - **VAL**: Để early stopping và tuning hyperparameters
> - **TEST**: Để đánh giá cuối cùng trước khi submit (internal test)
> - **Production**: Retrain trên 95% data (TRAIN+VAL+TEST) trước khi submit

### Step 3: HMM Regime Detection

```mermaid
flowchart TB
    HMM["GaussianHMM<br/>n_components=3"]
    
    HMM --> |"fit()"| FIT["Learn patterns<br/>from TRAIN only"]
    HMM --> |"predict()"| PRED["Label each day<br/>in TRAIN+VAL"]
    PRED --> LABELS["Regime Labels<br/>[0,1,2,0,1,1,2,...]"]
    LABELS --> LAST["regimes[-1]<br/>= Current Regime"]
    
    LAST --> |"Regime gì?"| SELECT["Chọn model<br/>tương ứng"]
```

> ⚠️ **QUAN TRỌNG:**
> - HMM fit CHỈ trên TRAIN → tránh data leakage
> - HMM predict trên TRAIN+VAL → để có regime labels
> - KHÔNG predict được trên TEST vì chưa có data!
> - `regimes[-1]` = regime ngày cuối → **GIẢ ĐỊNH** test cùng regime

### Step 4: Train Regime Models

```mermaid
flowchart TB
    LABELS["Regime Labels"] --> R0 & R1 & R2
    
    subgraph R0["Regime 0 - Stable"]
        D0["Data Regime 0"] --> M0["Model 0<br/>DLinear"]
    end
    
    subgraph R1["Regime 1 - Transition"]
        D1["Data Regime 1"] --> M1["Model 1<br/>DLinear"]
    end
    
    subgraph R2["Regime 2 - Volatile"]
        D2["Data Regime 2"] --> M2["Model 2<br/>DLinear"]
    end
```

> **Model học được pattern riêng cho từng Regime**

### Step 5: Grid Search & Validation

| Hyperparameter | Values |
|----------------|--------|
| `seq_len` | 60, 480 |
| `model` | Linear, DLinear |
| `variant` | Univariate, Multivariate |
| `n_regimes` | 3 |
| `regime_window` | 30, 60 |

**Evaluation:**
- Train models trên TRAIN
- Đánh giá MSE trên VAL
- Early stopping dựa trên VAL loss
- Chọn config có ValMSE thấp nhất

### Step 6: Production & Forecast

```mermaid
flowchart LR
    A["Best Config"] --> B["Retrain on 95%"]
    B --> C["Get regimes[-1]"]
    C --> D{"Current<br/>Regime?"}
    D --> |"0"| M0["Model 0"]
    D --> |"1"| M1["Model 1"]
    D --> |"2"| M2["Model 2"]
    M0 & M1 & M2 --> E["Predict 100 days"]
    E --> F["Inverse Transform"]
    F --> G[("submission.csv")]
```

## 📂 Cấu Trúc Dự Án

```
Project-6.1/
├── FPT_LTSF_GridSearch_Extended.ipynb   # Grid Search cho FPT
├── VIC_LTSF_GridSearch_Extended.ipynb   # Grid Search cho VIC
├── data/
│   ├── FPT_train.csv                    # Data FPT
│   ├── VIC_train.csv                    # Data VIC (train)
│   └── VIC_hidden_test.csv              # Data VIC (hidden test)
├── submissions/                          # Kết quả dự đoán
├── scripts/
│   └── debug_hmm_regimes.py             # Debug HMM visualization
└── README.md
```

## 🛠️ Cài Đặt

```bash
pip install torch pandas numpy scikit-learn hmmlearn matplotlib seaborn tqdm
```

## 📖 Hướng Dẫn Sử Dụng

### 1. Chuẩn bị data (nếu cần tách hidden test)

```bash
# Tách 100 ngày cuối làm hidden test
python -c "
import pandas as pd
df = pd.read_csv('data/VIC.csv')
df.iloc[:-100].to_csv('data/VIC_train.csv', index=False)
df.iloc[-100:].to_csv('data/VIC_hidden_test.csv', index=False)
"
```

### 2. Chạy Grid Search

Mở notebook tương ứng (FPT hoặc VIC) và chạy **Run All**.

Kết quả được lưu vào `submissions/` với format:
```
Sub_Multivariate_DLinear_HMM3W60_Seq60_MSE1234.csv
```

## 🧠 Chi Tiết Phương Pháp

### Feature Engineering

| Feature | Công thức | Công dụng |
|---------|-----------|-----------|
| **close_log** | `ln(close + 1)` | Stabilize variance |
| **HL_Spread** | `ln(high) - ln(low)` | Intraday volatility |
| **OC_Spread** | `ln(close) - ln(open)` | Price momentum |
| **returns** | `pct_change(close)` | For HMM |
| **volatility** | `rolling_std(returns)` | For HMM |
| **trend** | `pct_change(rolling_mean)` | For HMM |

### HMM Regime Detection

```python
class RegimeDetector:
    def fit(self, df_train):
        """Fit HMM trên train data only (avoid leakage)"""
        features = df[['returns', 'volatility', 'trend']]
        self.model.fit(features)
    
    def predict(self, df_full):
        """Predict regimes cho toàn bộ data"""
        return self.model.predict(features)
```

**Flow khi predict:**
1. `regimes = detector.predict(df)` → lấy regime cho mỗi ngày
2. `current_regime = regimes[-1]` → regime ngày cuối
3. `regime_model = models[current_regime]` → model tương ứng
4. `prediction = regime_model(last_sequence)` → kết quả

### Regime-Specific Training

```python
# Chia training data theo regime
for regime in [0, 1, 2]:
    mask = (regime_indices == regime)
    X_regime, y_regime = X_train[mask], y_train[mask]
    
    # Train model riêng cho regime này
    regime_models[regime] = train_model(X_regime, y_regime)
```

## 📊 Kết Quả & Insights

### Tại sao HMM Regime-Switching hiệu quả?

1. **Chuyên biệt hóa**: Thay vì 1 model học mọi pattern → nhiều models chuyên biệt
2. **Context-aware**: Prediction dựa trên điều kiện thị trường hiện tại
3. **Giảm noise**: Model chỉ học từ data có cùng đặc tính

### Grid Search Results (Example)

| Rank | Config | ValMSE | Hidden MSE |
|------|--------|--------|------------|
| 1 | Multi_DLinear_HMM3W60_Seq60 | 117 | 34 |
| 2 | Multi_DLinear_HMM3W30_Seq60 | 120 | 38 |
| 3 | Uni_DLinear_HMM3W60_Seq60 | 125 | 45 |

