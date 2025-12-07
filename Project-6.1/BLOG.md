# Dự Báo Giá Cổ Phiếu FPT: LTSF-Linear + HMM Regime-Switching

# I. Giới thiệu

Bài viết này trình bày giải pháp kỹ thuật cho cuộc thi **AIO-2025: LTSF-Linear Forecasting Challenge**. Mục tiêu của cuộc thi là dự báo giá đóng cửa (Close Price) của cổ phiếu FPT trong **100 ngày tiếp theo**.

Thay vì sử dụng các mô hình Deep Learning phức tạp (như LSTM, Transformer), giải pháp tập trung khai thác hiệu quả của các mô hình tuyến tính (Linear, NLinear, DLinear) kết hợp với kỹ thuật phân tích trạng thái thị trường (Market Regime).

## Thách thức chính

| Thách thức | Mô tả |
|------------|-------|
| **Long-term Forecasting** | Dự báo 100 ngày, không phải 1-7 ngày như thông thường |
| **Distribution Shift** | Giá cổ phiếu thay đổi range theo thời gian |
| **Market Regimes** | Thị trường có các trạng thái ẩn: ổn định, biến động, chuyển đổi |
| **Data Constraint** | Chỉ được dùng mỗi Data train, không có xài data ngoài |

## Giải pháp của nhóm

Nhóm kết hợp 3 kỹ thuật chính:

1.  **RevIN (Reversible Instance Normalization)**: Xử lý distribution shift bằng cách normalize đầu vào và denormalize đầu ra

2.  **HMM Regime Detection**: Sử dụng Hidden Markov Model để phát hiện trạng thái thị trường (Stable / Transition / Volatile)

3.  **Regime-Specific Models**: Train model riêng cho mỗi regime, dự báo dựa trên điều kiện thị trường hiện tại

## Kết quả đạt được hiện tại

| Method | Config | Private Score |
|--------|--------|---------------|
| **Univariate DLinear** | NoHMM + Seq480 | **28.98** |
| **Univariate Linear** | NoHMM + Seq480 | 39.81 |
| **Multivariate DLinear** | HMM + Seq60 | 47.60 |
| **Multivariate Linear** | HMM + Seq60 | 66.89 |


---

# II. Các thách thức

## 1. Long-term Forecasting

Dự báo 100 ngày là một thách thức lớn so với các bài toán dự báo ngắn hạn (1-7 ngày). Có hai phương pháp chính:

### Direct Forecasting (Multi-output)
- Train **nhiều model riêng biệt**, mỗi model dự báo một bước thời gian cụ thể
- Model 1 dự báo T+1, Model 2 dự báo T+2, ..., Model 100 dự báo T+100
- **Ưu điểm:** Không tích lũy error
- **Nhược điểm:** Cần train nhiều model, không capture được dependency giữa các bước

### Recursive Forecasting (Autoregressive)
- Train **một model duy nhất** dự báo bước tiếp theo
- Dùng prediction làm input để dự báo tiếp theo
- **Ưu điểm:** Một model, capture được dependency
- **Nhược điểm:** Error tích lũy theo thời gian

<p align="center">
  <img src="images/direct_vs_recursive.png" alt="Direct vs Recursive Forecasting" width="680">
  <br><em>Hình 1. So sánh Direct (trái) và Recursive (phải) Forecasting. (Nguồn: AI Viet Nam)</em>
</p>

**Trong project này:** Nhóm mình sử dụng **Direct Forecasting** - model dự báo trực tiếp 100 ngày một lần mà không cần recursive.

## 2. Distribution Shift

**Distribution shift** là hiện tượng phân phối dữ liệu thay đổi theo thời gian. Trong dữ liệu FPT, điều này thể hiện rõ ràng:

<p align="center">
  <img src="images/distribution_shift_fpt.png" alt="Distribution Shift FPT" width="680">
  <br><em>Hình 2. Distribution Shift trong dữ liệu FPT: Phân phối giá 2020-2021 hoàn toàn khác với 2023-2024.</em>
</p>

Như hình trên cho thấy:
- **Period 1 (2020-2021):** Giá dao động trong khoảng thấp
- **Period 2 (2023-2024):** Giá đã tăng lên mức cao hơn nhiều

Mặc dù **pattern biến động vẫn tương tự**, nhưng **scale đã thay đổi hoàn toàn**. Model học trên dữ liệu cũ sẽ dự báo sai scale nếu không xử lý.

**Giải pháp:** RevIN (Reversible Instance Normalization).

## 3. Market Regimes

Thị trường tài chính không hoạt động theo một quy luật duy nhất. Thay vào đó, nó chuyển đổi giữa các **trạng thái (regimes)** khác nhau:

- **Bull Market**: Xu hướng tăng mạnh, volatility thấp
- **Bear Market**: Xu hướng giảm, volatility cao
- **Sideways/Consolidation**: Đi ngang, không có xu hướng rõ ràng
- **Transition**: Giai đoạn chuyển đổi giữa các regime

<p align="center">
  <img src="images/market_regime_spx.png" alt="Market Regime Analysis" width="680">
  <br><em>Hình 3. Phân tích Market Regime trên S&P 500 (Nguồn: <a href="https://www.wallstreetcourier.com/spotlights/mrnl_sp-500-outlook-analyzing-the-current-market-regime-of-sp-500-spx/">Wall Street Courier</a>)</em>
</p>

**Vấn đề:** Một model duy nhất khó có thể học được tất cả patterns từ các regime khác nhau. Dữ liệu từ Bull Market có thể "nhiễu" việc học pattern của Bear Market và ngược lại.

**Giải pháp:** HMM Regime-Switching - phát hiện regime và train model chuyên biệt cho từng regime.

---

# III. Giải pháp kỹ thuật

## 1. RevIN - Reversible Instance Normalization

### 1.1 Ý tưởng

RevIN là kỹ thuật normalize dữ liệu **có thể đảo ngược**, được thiết kế đặc biệt cho time series với distribution shift. Ý tưởng chính:

1.  **Normalize input**: Chuẩn hóa chuỗi đầu vào về mean=0, std=1
2.  **Model học**: Model học patterns trên dữ liệu đã chuẩn hóa
3.  **Denormalize output**: Khôi phục lại scale gốc cho dự báo

<p align="center">
  <img src="images/fig1.gif" alt="RevIN Animation" width="500">
  <br><em>Hình 4. Tác dụng của RevIN. (Nguồn: <a href="https://github.com/ts-kim/RevIN/">RevIN GitHub</a>)</em>
</p>

### 1.2 Thuật toán

<p align="center">
  <img src="images/revin_algorithm.png" alt="RevIN Algorithm" width="600">
  <br><em>Hình 5. Thuật toán RevIN chi tiết. (Nguồn: <a href="https://github.com/ts-kim/RevIN/">RevIN GitHub</a>)</em>
</p>

### 1.3 Code Implementation

```python
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # Bước 4: Learnable parameters $\gamma$ và $\beta$
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            # Bước 1: Compute instance mean
            self.mean = x.mean(dim=1, keepdim=True).detach()
            # Bước 2: Compute instance variance  
            self.std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps).detach()
            # Bước 3: Normalize
            x = (x - self.mean) / self.std
            # Bước 4: Scale and shift với γ, β
            if self.affine:
                x = x * self.gamma + self.beta
            return x
            
        elif mode == 'denorm':
            # Bước 6: Reverse scale and shift
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            # Bước 7: Denormalize về scale gốc
            x = x * self.std + self.mean
            return x
```


### 1.4 Apply vào dữ liệu FPT

<p align="center">
  <img src="images/revin_fpt.png" alt="RevIN on FPT" width="680">
  <br><em>Hình 6. Áp dụng RevIN vào dữ liệu FPT.</em>
</p>

**Phân tích:**
- **Góc trên trái (Original):** Giá FPT có range thay đổi từ ~50 (2020) lên ~130 (2024)
- **Góc trên phải (After RevIN):** Sau normalize, giá dao động quanh 0 với std ≈ 1
- **Góc dưới trái (Distribution Original):** Phân phối lệch phải, nhiều peaks khác nhau
- **Góc dưới phải (Distribution After):** Phân phối gần chuẩn hơn, tập trung quanh 0

**Lợi ích:** Model không còn bị ảnh hưởng bởi sự thay đổi scale theo thời gian.

---

## 2. HMM Regime Detection

### 2.1 Hidden Markov Model

<p align="center">
  <img src="images/hmm_diagram.png" alt="HMM Diagram" width="500">
  <br><em>Hình 7. Minh họa Hidden Markov Model với 3 hidden states và 2 observations. (Nguồn: <a href="https://www.youtube.com/watch?v=RWkHJnFj5rY">YouTube</a>)</em>
</p>

HMM là mô hình xác suất trong đó:
- **Hidden states (Regimes):** Trạng thái ẩn mà ta không quan sát trực tiếp (ví dụ: mưa, mây, nắng)
- **Observations:** Các features ta đo được (ví dụ: buồn, vui)
- **Transition Matrix:** Xác suất chuyển giữa các trạng thái (các số 0.2, 0.3, 0.4, ...)

Trong bối cảnh thị trường chứng khoán:
- **Hidden states:** Bull Market, Bear Market, Sideways
- **Observations:** Returns, Volatility, Trend

### 2.2 Features cho HMM

Để HMM có thể phát hiện regime, ta cần cung cấp các features phản ánh "hành vi" thị trường:

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| **Returns** | $R_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$ | Tỷ suất sinh lời ngày, cho biết thị trường tăng hay giảm. $Close_t$: giá đóng cửa ngày $t$. |
| **Volatility** | $Vol_t = std(R_{t-9}, ..., R_t)$ | Độ biến động 10 ngày, cao = thị trường bất ổn |
| **Trend** | $Trend_t = \frac{MA_{10}(t) - MA_{10}(t-1)}{MA_{10}(t-1)}$ | Xu hướng trung bình động, cho biết trend tăng/giảm |

```python
# Tính toán features
df['returns'] = df['close'].pct_change().fillna(0)
df['volatility'] = df['returns'].rolling(window=10).std().fillna(0)
df['trend'] = df['close'].rolling(window=10).mean().pct_change().fillna(0)
```

<p align="center">
  <img src="images/hmm_features.png" alt="HMM Features" width="680">
  <br><em>Hình 8. Visualization các features cho HMM trên dữ liệu FPT.</em>
</p>

**Nhận xét từ hình:**
- **Returns:** Dao động quanh 0, có các spike lớn vào thời điểm biến động mạnh
- **Volatility:** Tăng cao vào các giai đoạn bất ổn (2020, 2022), thấp khi thị trường ổn định
- **Trend:** Cho thấy xu hướng tăng/giảm rõ ràng hơn so với returns

### 2.3 Regime Window

**Regime Window** là số ngày đầu tiên bị bỏ qua khi detect regimes:

```python
class RegimeDetector:
    def __init__(self, n_components=3, window=30):
        self.window = window
        
    def fit(self, df):
        # Bỏ qua `window` ngày đầu
        features = df[['returns', 'volatility', 'trend']].iloc[self.window:].values
        self.model.fit(features)
```

**Tại sao cần Regime Window?**
- Các features như `volatility` và `trend` cần rolling window để tính toán
- Những ngày đầu tiên có giá trị NaN hoặc không ổn định
- `window=30` → bỏ 30 ngày đầu, đảm bảo features đã ổn định

**Giá trị thường dùng:** 30, 60

<p align="center">
  <img src="images/regime_window.png" alt="Regime Window" width="680">
  <br><em>Hình 9. Regime Window: Bỏ qua 30 ngày đầu khi features chưa ổn định.</em>
</p>

### 2.4 Chọn số lượng Regimes

Câu hỏi: Nên dùng bao nhiêu regimes? 3? 4? 5?

#### N = 3 Regimes

<p align="center">
  <img src="images/hmm_3_regimes.png" alt="HMM 3 Regimes" width="680">
  <br><em>Hình 10. HMM với 3 Regimes trên dữ liệu FPT.</em>
</p>

**Phân tích:**
- **Regime 0 (Xanh lá):** Thường xuất hiện khi thị trường ổn định, volatility thấp
- **Regime 1 (Vàng):** Giai đoạn chuyển đổi, thường thấy trước khi thị trường đổi hướng
- **Regime 2 (Đỏ):** Thị trường biến động mạnh, có thể là crash hoặc rally mạnh

**Nhận xét:** Phân chia 3 regime khá rõ ràng, mỗi regime có đủ samples để train model.

#### N = 4 Regimes

<p align="center">
  <img src="images/hmm_4_regimes.png" alt="HMM 4 Regimes" width="680">
  <br><em>Hình 11. HMM với 4 Regimes trên dữ liệu FPT.</em>
</p>

**Phân tích:**
- Phân chia chi tiết hơn với 4 trạng thái
- Một số regime có thể có ít samples, gây khó khăn cho training
- Có thể bắt được nhiều chi tiết hơn, nhưng cũng dễ overfit

**Trade-off:**

| N Regimes | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **N = 2** | Đơn giản, nhiều samples/regime | Quá thô, bỏ sót chi tiết |
| **N = 3** | Cân bằng, phổ biến | Có thể không đủ chi tiết |
| **N = 4+** | Chi tiết hơn | Ít samples/regime, dễ overfit |

**Trong project này:** Nhóm chọn **N = 3** vì:
1.  Đủ chi tiết để phân biệt bull/bear/transition
2.  Mỗi regime có đủ samples để train

### 2.5 Lưu ý quan trọng

> **Giải pháp:** Giả định regime hiện tại (`regimes[-1]`) tiếp tục trong 100 ngày forecast.

---

## 3. LTSF-Linear Models

### 3.1 RLinear (Linear + RevIN)

<p align="center">
  <img src="images/RLinear.png" alt="RLinear Architecture" width="500">
  <br><em>Hình 12. Kiến trúc Linear + RevIN: RevIN → Linear → Denormalize.</em>
</p>

**Kiến trúc:**
1.  **RevIN Normalize**: Chuẩn hóa input về mean=0, std=1
2.  **Linear Layer**: Một lớp fully-connected ánh xạ từ `seq_len` → `pred_len`
3.  **Denormalize**: Khôi phục scale gốc cho output

**Công thức:**
$$\hat{y} = W \cdot x_{norm} + b$$
$$y = \hat{y} \cdot \sigma + \mu$$

Trong đó $W \in \mathbb{R}^{pred\_len \times seq\_len}$ là ma trận trọng số.

```python
class Linear(nn.Module):
    def __init__(self, seq_len, pred_len, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # RevIN normalize
        x = self.revin(x, 'norm')
        
        # Linear projection
        out = self.linear(x[:, :, 0])  # Univariate: chỉ dùng close
        
        # RevIN denormalize
        out = self.revin(out.unsqueeze(-1), 'denorm').squeeze(-1)
        return out
```

### 3.2 RDLinear (DLinear + RevIN)

<p align="center">
  <img src="images/RDLinear.png" alt="RDLinear Architecture" width="600">
  <br><em>Hình 13. Kiến trúc DLinear + RevIN: Decomposition thành Trend + Seasonal.</em>
</p>

**Ý tưởng:** Tách chuỗi thời gian thành 2 thành phần:
-   **Trend**: Xu hướng dài hạn (tính bằng Moving Average)
-   **Seasonal**: Biến động ngắn hạn (phần còn lại)

**Công thức:**
$$x_{trend} = \text{MovingAvg}(x, kernel)$$
$$x_{seasonal} = x - x_{trend}$$
$$\hat{y} = W_t \cdot x_{trend} + W_s \cdot x_{seasonal}$$

```python
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, kernel_size=25):
        super().__init__()
        self.revin = RevIN(num_features)
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # RevIN normalize
        x = self.revin(x, 'norm')
        x_in = x[:, :, 0]  # Univariate
        
        # Decomposition
        trend = self.moving_avg(x_in.unsqueeze(1)).squeeze(1)
        seasonal = x_in - trend
        
        # Separate linear projections
        out = self.linear_trend(trend) + self.linear_seasonal(seasonal)
        
        # RevIN denormalize
        out = self.revin(out.unsqueeze(-1), 'denorm').squeeze(-1)
        return out
```

**Tại sao DLinear tốt hơn?**
-   Trend và Seasonal có patterns khác nhau → cần weights khác nhau
-   Linear đơn phải học cả 2 patterns cùng lúc → khó hơn

---

## 4. Regime-Specific Training

### 4.1 Ý tưởng

Thay vì train **một model duy nhất** trên toàn bộ dữ liệu, ta:
1.  Dùng **HMM để phân cụm** dữ liệu thành các regimes (trạng thái thị trường ẩn)
2.  **Train một model riêng** trên dữ liệu của từng regime
3.  Khi forecast: xác định **regime hiện tại** → chọn model đó → predict

### 4.2 Code

```python
# === TRAINING ===
# 1. Fit HMM trên TRAIN data
hmm = GaussianHMM(n_components=3)
hmm.fit(train_features)  # features = [returns, volatility, trend]

# 2. Predict regimes cho TRAIN + VAL
regimes = hmm.predict(trainval_features)

# 3. Train model riêng cho mỗi regime
models = {}
for r in [0, 1, 2]:
    mask = (regimes == r)
    X_r, y_r = X_trainval[mask], y_trainval[mask]
    
    models[r] = DLinear(seq_len, pred_len, num_features)
    train(models[r], X_r, y_r)

# === PREDICTION ===
# 4. Lấy regime cuối cùng
current_regime = regimes[-1]

# 5. Dùng model tương ứng để predict
prediction = models[current_regime](last_sequence)
```

### 4.3 Tại sao hiệu quả?

| Cách tiếp cận | Vấn đề |
|---------------|--------|
| **1 model cho tất cả** | Phải học cùng lúc pattern của bull, bear, sideways → confused |
| **Model riêng theo regime** | Mỗi model chỉ tập trung học pattern của 1 regime → specialized |

**Ví dụ:**
-   **Regime 0 (Stable):** Model học pattern ổn định, volatility thấp
-   **Regime 1 (Transition):** Model học các dấu hiệu đổi hướng
-   **Regime 2 (Volatile):** Model học cách xử lý biến động mạnh

---

# IV. Luồng xử lý

## Tổng quan Pipeline

```mermaid
flowchart TB
    subgraph STEP1["STEP 1: DATA LOADING"]
        A[("Raw Data<br/>1149 x 6")]
    end
    
    subgraph STEP2["STEP 2: DATA SPLITTING"]
        direction LR
        B1["TRAIN<br/>839 days"]
        B2["VAL<br/>210 days"]
        B3["TEST<br/>100 days"]
    end
    
    subgraph STEP3["STEP 3: FEATURE ENGINEERING"]
        direction LR
        C1["Log Transform"]
        C2["Spread Features"]
        C3["HMM Features"]
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
        H[("submission.csv<br/>100 days forecast")]
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

---

## 1. Data Loading

Load dữ liệu OHLCV (Open, High, Low, Close, Volume) từ file CSV:

```python
df = pd.read_csv('data/FPT_train.csv')
# Columns: time, open, high, low, close, volume
# 1149 rows (days)
```

---

## 2. Feature Engineering

```mermaid
flowchart LR
    A["Raw Data<br/>OHLCV"] --> B["Log Transform"]
    A --> C["Spread Features"]
    B --> D["close_log<br/>volume_log"]
    C --> E["HL_Spread<br/>OC_Spread"]
    D & E --> F["HMM Features"]
    F --> G["returns<br/>volatility<br/>trend"]
```

### 2.1 Log Transform

Áp dụng log transform cho `close` và `volume` để ổn định phương sai:

```python
df['close_log'] = np.log(df['close'])
df['volume_log'] = np.log(df['volume'] + 1)
```

**Tại sao?** Dữ liệu tài chính thường có phân phối lệch phải. Log transform giúp:
-   Ổn định phương sai
-   Dễ học pattern hơn

### 2.2 Spread Features

<p align="center">
  <img src="images/spread_features.png" alt="Spread Features" width="680">
  <br><em>Hình 14. Visualization các Spread Features trên dữ liệu FPT.</em>
</p>

**HL_Spread (High-Low Spread):**
$$HL\_Spread = \frac{High - Low}{Low} \times 100\%$$

-   Đo **độ biến động trong ngày**
-   Cao → thị trường biến động mạnh
-   Thấp → thị trường ổn định

**OC_Spread (Open-Close Spread):**
$$OC\_Spread = \frac{Close - Open}{Open} \times 100\%$$

-   Đo **xu hướng trong ngày**
-   Dương (xanh) → ngày tăng
-   Âm (đỏ) → ngày giảm

trong đó: $High, Low, Open, Close$ là giá cao nhất, thấp nhất, mở cửa và đóng cửa trong ngày.

```python
df['HL_Spread'] = (df['high'] - df['low']) / df['low']
df['OC_Spread'] = (df['close'] - df['open']) / df['open']
```

### 2.3 HMM Features

Như đã trình bày ở phần III.2:

```python
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=10).std()
df['trend'] = df['close'].rolling(window=10).mean().pct_change()
```

---

## 3. Data Splitting

```mermaid
pie title Data Split (1149 days)
    "TRAIN (839 - 73%)" : 839
    "VAL (210 - 18%)" : 210
    "TEST (100 - 9%)" : 100
```

| Split | Days | Mục đích |
|-------|------|----------|
| **TRAIN** | 839 (73%) | Train model |
| **VAL** | 210 (18%) | Early stopping, tuning |
| **TEST** | 100 (9%) | Đánh giá cuối cùng |

---

## 4. HMM Regime Detection

```mermaid
flowchart TB
    HMM["GaussianHMM<br/>n_components=3"]
    
    HMM --> |"fit()"| FIT["Learn patterns<br/>from TRAIN only"]
    HMM --> |"predict()"| PRED["Label each day<br/>in TRAIN+VAL"]
    PRED --> LABELS["Regime Labels<br/>[0,1,2,0,1,1,2,...]"]
    LABELS --> LAST["regimes[-1]<br/>= Current Regime"]
    
    LAST --> |"Regime gì?"| SELECT["Chọn model<br/>tương ứng"]
```
<p align="center">
  <img src="images/regimes-1.png" alt="Minh họa regimes[-1]" width="680">
  <br><em>Hình 15. Minh họa việc lấy regime cuối cùng (regimes[-1]).</em>
</p>

> **LƯU Ý QUAN TRỌNG: Tránh Data Leakage**
> 
> -   **fit()** CHỈ trên TRAIN → để học patterns
> -   **predict()** trên TRAIN+VAL → để có regime labels cho cả 2
> -   KHÔNG predict được trên TEST vì chưa có data!

```python
# Fit HMM CHỈ trên TRAIN
hmm = RegimeDetector(n_components=3)
hmm.fit(train_df)

# Predict trên TRAIN + VAL
regimes = hmm.predict(trainval_df)
```

---

## 5. Model Training (Per Regime)

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

Với mỗi regime, train một model riêng:

```python
for r in [0, 1, 2]:
    mask = (regimes == r)
    X_r, y_r = X_trainval[mask], y_trainval[mask]
    models[r] = DLinear(seq_len, pred_len, num_features)
    train(models[r], X_r, y_r)
```

---

## 6. Evaluation on TEST

Đánh giá model trên TEST set để kiểm tra:

```python
# Lấy regime cuối của TRAINVAL
test_regime = regimes[-1]

# Dùng model tương ứng để predict
predictions = models[test_regime](X_test)

# Tính MSE
test_mse = ((predictions - y_test) ** 2).mean()
```

**Mục đích:** Đảm bảo pipeline hoạt động tốt trước khi submit.

---

## 7. Production & Submission

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

### 7.1 Retrain for Production

Sau khi đã validate xong, retrain trên **95% toàn bộ data**:

```python
# Chia lại data: 95% train, 5% kept for regime detection
production_df = full_df.iloc[:int(len(full_df) * 0.95)]

# Fit HMM lại trên production data
hmm.fit(production_df)
regimes = hmm.predict(production_df)
```

### 7.2 Select Model by Current Regime

```python
# Regime cuối cùng = "tâm lý thị trường" hiện tại
current_regime = regimes[-1]
print(f"Current market regime: {current_regime}")
```

### 7.3 Final Prediction & Submission

```python
# Chọn model tương ứng với current regime
final_model = models[current_regime]

# Predict 100 ngày
last_sequence = get_last_sequence(production_df)
predictions = final_model(last_sequence)

# Inverse transform (nếu dùng log)
predictions = np.exp(predictions)

# Tạo submission
submission = pd.DataFrame({
    'row_id': range(len(predictions)),
    'close': predictions
})
submission.to_csv('submission.csv', index=False)
```

---

# V. Kết quả đánh giá

Nhóm sử dụng kết quả từ hệ thống Kaggle để có đánh giá khách quan nhất dựa trên **Private Leaderboard**.

## Bảng kết quả (Kaggle Leaderboard)

> **LƯU Ý QUAN TRỌNG VỀ ĐIỂM SỐ:**
> Trong quá trình thi, nhóm từng đạt được mức **Private Score 14.35** (như hình Kaggle cũ). Tuy nhiên, sau khi kiểm tra kỹ lưỡng, nhóm phát hiện đó là kết quả của việc **Data Leakage** (do sơ suất trong khâu xử lý data pipeline). 
> 
> Sau khi fix lỗi và retrain lại pipeline chuẩn, số điểm ổn định (stable score) mà nhóm đạt được là **28.98**. Đây mới là kết quả thực sự phản ánh hiệu năng của giải pháp. Nhóm quyết định trung thực với kết quả này thay vì "ăn may".

### Top 4 Models tốt nhất

| # | Model | Type | Config | Private Score |
|---|-------|------|--------|---------------|
| 1 | **Univariate DLinear** | Đơn biến | Seq480 (NoHMM) | **28.9824** |
| 2 | **Univariate Linear** | Đơn biến | Seq480 (NoHMM) | 39.8063 |
| 3 | **Multivariate DLinear** | Đa biến | Seq60 (HMM) | 47.6060 |
| 4 | **Multivariate Linear** | Đa biến | Seq60 (HMM) | 66.8885 |

### So sánh trực quan

<p align="center">
  <img src="images/four_models_grid.png" alt="Four Models Grid" width="800">
  <br><em>Hình 16. Dự báo chi tiết của từng model.</em>
</p>

<p align="center">
  <img src="images/four_models_combined.png" alt="Four Models Combined" width="800">
  <br><em>Hình 17. So sánh tổng hợp: Univariate (Warm colors) vs Multivariate (Cool colors).</em>
</p>

## Phân tích kết quả chi tiết

### 1. Cuộc chiến Sequence Length: Seq480 (Long) vs Seq60 (Short)

<p align="center">
  <img src="images/analysis_seqlen.png" alt="SeqLen Analysis" width="700">
  <br><em>Hình 18. Impact của Sequence Length.</em>
</p>

-   **Univariate Seq480 (Đỏ - Best):** Nhờ nhìn được lịch sử dài (480 ngày ~ 2 năm), model nắm bắt được **xu hướng dài hạn** (long-term trend) của FPT. Đường dự báo đầm, chắc chắn và bám sát xu hướng tăng trưởng.
-   **Univariate Seq60 (Cam - Overfit):** Chỉ nhìn 60 ngày (~3 tháng), model bị "cuốn" theo các biến động ngắn hạn (noise). Kết quả là Private Score cực tệ (~203) do overfitting vào dữ liệu train gần nhất.

> **Kết luận:** Với bài toán dự báo dài hạn (100 ngày), việc sử dụng **input sequence đủ dài** (Look-back window lớn) quan trọng hơn nhiều so với việc dùng model phức tạp.

### 2. Cuộc chiến HMM: Có HMM vs Không HMM

<p align="center">
  <img src="images/analysis_hmm.png" alt="HMM Analysis" width="700">
  <br><em>Hình 19. Impact của HMM trên Multivariate Models.</em>
</p>

-   **Multivariate HMM (Xanh - Stable):** Khi dùng nhiều biến (đa biến), dữ liệu trở nên rất nhiễu. HMM giúp **phân cụm nhiễu** bằng cách chia thị trường thành các regimes (Stable vs Volatile). Nhờ đó forecast (đường xanh) ổn định hơn, Private Score 47.60.
-   **Multivariate NoHMM (Xám - Volatile):** Nếu không có HMM, model đa biến bị nhiễu loạn bởi các tín hiệu conflicting từ nhiều features. Kết quả dự báo (đường xám) đi lệch hẳn, Private Score tệ (~249).

> **Kết luận:** Nếu dùng Multivariate, **HMM là bắt buộc** để kiểm soát nhiễu. Tuy nhiên, ngay cả khi có HMM, performance vẫn thua Univariate đơn giản.

### 3. Univariate vs Multivariate: Đơn giản là nhất?

Tại sao Univariate (28.98) lại thắng Multivariate (47.60)?

1.  **Noise vs Signal:** FPT là mã cổ phiếu blue-chip, biến động khá tuân theo quy luật cung cầu dài hạn. Các biến thêm vào (Open, High, Low, Volume) trong bài toán forecast 100 ngày dường như đóng vai trò là **Noise** nhiều hơn là Signal hữu ích.
2.  **Robustness:** Model đơn biến ít tham số hơn, khó overfit hơn. Trong bối cảnh dự báo dài hạn, sự ổn định (robustness) quan trọng hơn sự phức tạp.

### Kết luận cuối cùng

Mặc dù giải pháp **Multivariate + HMM** (được thiết kế công phu) rất hứa hẹn về mặt kỹ thuật, nhưng thực tế chứng minh **Univariate DLinear với long sequence** (đơn giản, nhìn xa) mới là "chân ái" cho tập dữ liệu này.

> **Bài học rút ra:** "Simple is Best". Đôi khi việc hiểu data (xử lý trend dài hạn) quan trọng hơn việc áp dụng model fancy. 

## Kết luận

| Insight | Giải thích |
|---------|------------|
| **Univariate > Multivariate** | Ít noise hơn, tập trung vào target |
| **Seq480 > Seq60** | Nhìn trend dài hạn, tránh overfitting |
| **DLinear > Linear** | Trend-Seasonal decomposition giúp ích |
| **HMM giúp Multivariate** | Phân cụm data giảm conflict |
| **TrainMSE cao ≠ xấu** | Có thể là dấu hiệu của generalization tốt |

---

# Bonus: Kết quả trên VIC

Nhóm cũng áp dụng pipeline tương tự cho cổ phiếu **VIC (Vingroup)**:

## Dữ liệu VIC

<p align="center">
  <img src="images/vic_train_vs_hidden.png" alt="VIC Train vs Test Data" width="680">
  <br><em>Hình 20. Dữ liệu VIC: Train (xanh) vs Test Data (cam).</em>
</p>

**Đặc điểm VIC khác FPT:**
-   Downtrend dài từ 2019-2023 (~120 → ~40)
-   Test Data có rally mạnh (~40 → ~120)
-   **Thách thức lớn:** Model train trên downtrend, phải predict uptrend!

## So sánh các predictions

<p align="center">
  <img src="images/vic_predictions.png" alt="VIC Predictions" width="680">
  <br><em>Hình 21. So sánh predictions của các models trên VIC test data.</em>
</p>

**Nhận xét:**
  - Không có thông tin gì về catalyst (news, events) gây rally
  - **Regime shift** từ bearish → bullish không được capture

**Bài học:**
> LTSF-Linear (và các technical models nói chung) chỉ có thể dự đoán dựa trên **historical patterns**. Khi có **regime change** mạnh (fundamental shifts), models sẽ khó predict chính xác.

---

# VI. Kết luận

## Tóm tắt

Trong project này, nhóm đã:

1. **Áp dụng LTSF-Linear** cho bài toán dự đoán giá cổ phiếu
2. **Sử dụng RevIN** để xử lý distribution shift
3. **Kết hợp HMM Regime-Switching** để phân cụm market states
4. **Grid Search** để tìm config tốt nhất

## Findings chính

| Finding | Giải thích |
|---------|------------|
| **Univariate DLinear Seq480 = Best** | Đơn giản, nhìn trend dài hạn |
| **HMM giúp Multivariate** | Phân cụm giảm conflict |
| **TrainMSE không phải tất cả** | Cần đánh giá trên unseen data |

## Hạn chế

- **Regime assumption:** Giả định regime cuối cùng tiếp tục trong 100 ngày
- **No external factors:** Không có news, events, macro data
- **Linear models:** Có thể miss non-linear patterns

## Hướng phát triển

1. Thêm **external features** (sentiment, news)
2. Thử **ensemble** multiple regimes
3. Combine với **Transformer-based** models

---

**Cảm ơn bạn đã đọc!**

### Tags
Time Series Forecasting, Linear Models, HMM, Regime Switching, Stock Prediction, FPT, Kaggle

