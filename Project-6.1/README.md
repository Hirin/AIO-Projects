# 📈 FPT Stock Prediction: LTSF-Linear + HMM Regime-Switching

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📖 Overview

This project predicts the closing price of **FPT stock** for the next **100 days** using **LTSF-Linear** models combined with **HMM Regime-Switching**.

> For a detailed technical write-up, see [BLOG.md](BLOG.md).

## 🏆 Results (Kaggle Private Leaderboard)

| # | Model | Config | Private Score |
|---|-------|--------|---------------|
| 1 | **Univariate DLinear** | Seq480, NoHMM | **28.98** |
| 2 | Univariate Linear | Seq480, NoHMM | 39.81 |
| 3 | Multivariate DLinear | Seq60, HMM | 47.60 |
| 4 | Multivariate Linear | Seq60, HMM | 66.89 |

## 💡 Key Insights

1.  **Longer Lookback Window is Better:** Seq480 (≈2 years) significantly outperforms Seq60 (≈3 months) for long-term forecasting.
2.  **Univariate > Multivariate:** Using only `close` price is more stable. Adding more features introduces noise.
3.  **HMM is Essential for Multivariate:** HMM reduces MSE from ~249 to ~47 for multivariate models by separating market regimes.

## 🔧 Core Techniques

| Technique | Purpose |
|-----------|---------|
| **RevIN** | Handles distribution shift by normalizing inputs and denormalizing outputs |
| **HMM Regime Detection** | Identifies hidden market states (Stable, Transition, Volatile) |
| **Regime-Specific Training** | Trains a specialized model for each regime |

## 📂 Project Structure

```
Project-6.1/
├── BLOG.md                               # Detailed technical blog post
├── FPT_LTSF_GridSearch_Extended.ipynb    # Main experiment notebook
├── data/
│   └── FPT_train.csv                     # Training data
└── submissions/                          # Submission files
```

## 🛠️ Installation

```bash
pip install torch pandas numpy scikit-learn hmmlearn matplotlib seaborn tqdm
```

## 🚀 Quick Start

1.  Open `FPT_LTSF_GridSearch_Extended.ipynb`.
2.  Run all cells.
3.  Results are saved in `submissions/`.

## 📊 Pipeline Overview

```
Data Loading → Feature Engineering → Data Splitting (73/18/9)
     ↓
HMM Regime Detection (fit on TRAIN, predict on TRAIN+VAL)
     ↓
Train Model per Regime → Evaluate on VAL/TEST
     ↓
Retrain on 95% → Select model by regimes[-1] → Predict 100 days
```

