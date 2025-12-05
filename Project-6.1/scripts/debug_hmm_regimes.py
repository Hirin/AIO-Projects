"""
Debug HMM Regime Clustering for FPT Stock Prediction

This script creates clear visualizations to understand:
1. How HMM regime splits the data
2. Whether data loses continuity when split by regime
3. Which regime is used for hidden test prediction
4. Why data leakage might help (or not hurt) performance
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
DATA_PATH = 'data/FPT_train.csv'
OUTPUT_DIR = 'results/debug_hmm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HMM Parameters (same as notebook)
N_REGIMES = 3
REGIME_WINDOW = 30
SEQ_LEN = 60  # Example sequence length
PRED_LEN = 100


def load_and_prepare_data():
    """Load data and create features for HMM"""
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Create HMM features
    df['returns'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(window=10).std().fillna(0)
    df['trend'] = df['close'].rolling(window=10).mean().pct_change().fillna(0)
    
    return df


def fit_hmm_full(df, n_components=3, window=30):
    """Fit HMM on FULL data (current approach with data leakage)"""
    features = df[['returns', 'volatility', 'trend']].iloc[window:].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = GaussianHMM(n_components=n_components, covariance_type="full", 
                        n_iter=100, random_state=42)
    model.fit(features_scaled)
    states = model.predict(features_scaled)
    
    full_states = np.concatenate([np.zeros(window) - 1, states])
    return full_states.astype(int), model, scaler


def fit_hmm_train_only(df, train_size, n_components=3, window=30):
    """Fit HMM only on TRAIN data (correct approach without leakage)"""
    train_df = df.iloc[:train_size]
    features = train_df[['returns', 'volatility', 'trend']].iloc[window:].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = GaussianHMM(n_components=n_components, covariance_type="full", 
                        n_iter=100, random_state=42)
    model.fit(features_scaled)
    
    # Predict on full data using train-fitted model
    full_features = df[['returns', 'volatility', 'trend']].iloc[window:].values
    full_features_scaled = scaler.transform(full_features)
    states = model.predict(full_features_scaled)
    
    full_states = np.concatenate([np.zeros(window) - 1, states])
    return full_states.astype(int), model, scaler


def plot_1_regime_timeline(df, regimes, output_path):
    """Plot 1: Price with regime coloring"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    regime_names = ['Regime 0 (Stable)', 'Regime 1 (Transition)', 'Regime 2 (Volatile)']
    
    # Plot price line
    ax.plot(df['time'], df['close'], 'k-', linewidth=0.8, alpha=0.8)
    
    # Color background by regime
    for i in range(len(df) - 1):
        if regimes[i] >= 0:
            ax.axvspan(df['time'].iloc[i], df['time'].iloc[i+1], 
                      alpha=0.3, color=colors[regimes[i]])
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.5, label=regime_names[i]) 
                      for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title('FPT Stock Price with HMM Regimes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_2_regime_data_splitting(df, regimes, train_size, seq_len, output_path):
    """Plot 2: How regime splits training data into subsets"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    # Calculate training window indices (like in sliding window)
    n_samples = train_size - seq_len - PRED_LEN + 1
    sample_regime_indices = []
    for i in range(n_samples):
        # Regime of the last day of each training window
        r_idx = i + seq_len - 1
        sample_regime_indices.append(regimes[r_idx] if r_idx < len(regimes) else -1)
    sample_regime_indices = np.array(sample_regime_indices)
    
    # Plot each regime's training samples
    for regime_id in range(3):
        ax = axes[regime_id]
        mask = (sample_regime_indices == regime_id)
        sample_indices = np.where(mask)[0]
        
        # Plot timeline showing which samples belong to this regime
        ax.scatter(sample_indices, np.ones(len(sample_indices)) * 0.5, 
                  c=colors[regime_id], s=10, alpha=0.6)
        
        # Mark gaps in continuity
        if len(sample_indices) > 1:
            gaps = np.diff(sample_indices)
            large_gaps = np.where(gaps > 10)[0]
            for gap_idx in large_gaps:
                ax.axvline(x=sample_indices[gap_idx], color='red', 
                          linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        ax.set_title(f'Regime {regime_id}: {mask.sum()} samples ({mask.sum()/len(mask)*100:.1f}%)', 
                    fontsize=12, color=colors[regime_id])
        ax.set_xlabel('Training Sample Index')
        ax.set_yticks([])
    
    fig.suptitle('Training Data Split by Regime\n(Red dashed lines = gaps > 10 samples, potential discontinuity)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")
    
    # Print continuity analysis
    print("\n📊 REGIME CONTINUITY ANALYSIS:")
    for regime_id in range(3):
        mask = (sample_regime_indices == regime_id)
        sample_indices = np.where(mask)[0]
        if len(sample_indices) > 1:
            gaps = np.diff(sample_indices)
            large_gaps = (gaps > 10).sum()
            max_consecutive = max(len(list(g)) for k, g in 
                                 __import__('itertools').groupby(mask) if k)
            print(f"  Regime {regime_id}: {len(sample_indices)} samples, "
                  f"{large_gaps} large gaps, max consecutive: {max_consecutive}")


def plot_3_hidden_test_prediction_mechanism(df, regimes, train_size, seq_len, output_path):
    """Plot 3: Which regime is used for hidden test prediction"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    # Top: Show the last portion of data and highlight current regime
    ax1 = axes[0]
    last_n = 200  # Show last 200 days
    plot_df = df.iloc[-last_n:]
    plot_regimes = regimes[-last_n:]
    
    ax1.plot(range(len(plot_df)), plot_df['close'].values, 'k-', linewidth=1)
    
    for i in range(len(plot_df) - 1):
        if plot_regimes[i] >= 0:
            ax1.axvspan(i, i+1, alpha=0.3, color=colors[plot_regimes[i]])
    
    # Mark the LAST DAY with a big marker
    current_regime = regimes[-1]
    ax1.scatter([len(plot_df)-1], [plot_df['close'].values[-1]], 
               c=colors[current_regime], s=200, marker='*', edgecolors='black',
               zorder=5, label=f'Last Day → Regime {current_regime}')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_title('Last 200 Days: Current Regime for Prediction', fontsize=12)
    ax1.set_xlabel('Days from end')
    ax1.set_ylabel('Close Price')
    
    # Bottom: Show code logic and statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    # Calculate statistics
    prod_train_size = int(len(df) * 0.95)
    prod_regime_indices = []
    for i in range(prod_train_size - seq_len - PRED_LEN + 1):
        r_idx = i + seq_len - 1
        prod_regime_indices.append(regimes[r_idx] if r_idx < len(regimes) else -1)
    prod_regime_indices = np.array(prod_regime_indices)
    
    regime_counts = [(prod_regime_indices == r).sum() for r in range(3)]
    current_regime_count = regime_counts[current_regime]
    
    code_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  HIDDEN TEST PREDICTION MECHANISM (Code Flow)                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. current_regime = regimes[-1]  # Regime của ngày CUỐI CÙNG            ║
║     → current_regime = {current_regime}                                              ║
║                                                                          ║
║  2. mask = (prod_regime_indices == current_regime)                       ║
║     → Số mẫu training thuộc regime {current_regime}: {current_regime_count} mẫu                   ║
║                                                                          ║
║  3. if mask.sum() > 30:   # Đủ dữ liệu → Train regime-specific model     ║
║         regime_model = train_model(X_regime, y_regime)                   ║
║         pred = regime_model(last_sequence)                               ║
║     → Điều kiện thỏa mãn: {current_regime_count} > 30 = {current_regime_count > 30}                             ║
║                                                                          ║
║  CONCLUSION: Model dự đoán hidden test được train ONLY trên Regime {current_regime}    ║
╚══════════════════════════════════════════════════════════════════════════╝

REGIME TRAINING DATA DISTRIBUTION:
  • Regime 0: {regime_counts[0]} samples ({regime_counts[0]/sum(regime_counts)*100:.1f}%)
  • Regime 1: {regime_counts[1]} samples ({regime_counts[1]/sum(regime_counts)*100:.1f}%)  
  • Regime 2: {regime_counts[2]} samples ({regime_counts[2]/sum(regime_counts)*100:.1f}%)
  
→ Hidden test prediction uses Regime {current_regime} with {current_regime_count} training samples
"""
    ax2.text(0.02, 0.5, code_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Hidden Test Prediction: Which Regime is Used?', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")
    
    return current_regime, current_regime_count


def plot_4_regime_statistics(df, regimes, output_path):
    """Plot 4: Statistics per regime"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    regime_data = {i: [] for i in range(3)}
    
    for i, r in enumerate(regimes):
        if r >= 0 and i < len(df):
            regime_data[r].append(df['returns'].iloc[i])
    
    # Plot 1: Mean Return per Regime
    means = [np.mean(regime_data[r])*100 if regime_data[r] else 0 for r in range(3)]
    axes[0].bar(range(3), means, color=colors)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Mean Daily Return (%)', fontsize=12)
    axes[0].set_xlabel('Regime')
    axes[0].set_xticks(range(3))
    for i, v in enumerate(means):
        axes[0].text(i, v, f'{v:.3f}%', ha='center', va='bottom' if v >= 0 else 'top')
    
    # Plot 2: Volatility per Regime
    stds = [np.std(regime_data[r])*100 if regime_data[r] else 0 for r in range(3)]
    axes[1].bar(range(3), stds, color=colors)
    axes[1].set_title('Daily Volatility (%)', fontsize=12)
    axes[1].set_xlabel('Regime')
    axes[1].set_xticks(range(3))
    for i, v in enumerate(stds):
        axes[1].text(i, v, f'{v:.3f}%', ha='center', va='bottom')
    
    # Plot 3: Sample Count per Regime
    counts = [len(regime_data[r]) for r in range(3)]
    axes[2].bar(range(3), counts, color=colors)
    axes[2].set_title('Sample Count', fontsize=12)
    axes[2].set_xlabel('Regime')
    axes[2].set_xticks(range(3))
    for i, v in enumerate(counts):
        axes[2].text(i, v, str(v), ha='center', va='bottom')
    
    fig.suptitle('Regime Characteristics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")
    
    # Print summary
    print("\n📊 REGIME CHARACTERISTICS:")
    for r in range(3):
        print(f"  Regime {r}: count={counts[r]}, mean_return={means[r]:.4f}%, volatility={stds[r]:.4f}%")


def plot_5_leakage_comparison(df, regimes_full, regimes_train, train_size, output_path):
    """Plot 5: Compare full-fit vs train-only fit regimes"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    
    # Top: Full-fit regimes
    ax1 = axes[0]
    for i in range(len(df) - 1):
        if regimes_full[i] >= 0:
            ax1.axvspan(i, i+1, alpha=0.4, color=colors[regimes_full[i]])
    ax1.axvline(x=train_size, color='blue', linestyle='--', linewidth=2, label='Train/Val Split')
    ax1.set_title(f'Full-Fit Regimes (Data Leakage) - Last day: Regime {regimes_full[-1]}', fontsize=12)
    ax1.set_xlim(0, len(df))
    ax1.legend()
    
    # Middle: Train-only fit regimes
    ax2 = axes[1]
    for i in range(len(df) - 1):
        if regimes_train[i] >= 0:
            ax2.axvspan(i, i+1, alpha=0.4, color=colors[regimes_train[i]])
    ax2.axvline(x=train_size, color='blue', linestyle='--', linewidth=2, label='Train/Val Split')
    ax2.set_title(f'Train-Only Fit Regimes (No Leakage) - Last day: Regime {regimes_train[-1]}', fontsize=12)
    ax2.set_xlim(0, len(df))
    ax2.legend()
    
    # Bottom: Difference
    ax3 = axes[2]
    diff = (regimes_full != regimes_train).astype(float)
    diff[regimes_full < 0] = 0  # Ignore initial padding
    diff[regimes_train < 0] = 0
    ax3.bar(range(len(diff)), diff, color='red', alpha=0.5, width=1)
    ax3.axvline(x=train_size, color='blue', linestyle='--', linewidth=2, label='Train/Val Split')
    ax3.set_title(f'Differences (Red = different regime) - Total differences: {int(diff.sum())} days', fontsize=12)
    ax3.set_xlim(0, len(df))
    ax3.set_xlabel('Day Index')
    ax3.legend()
    
    fig.suptitle('Data Leakage Impact: Full-Fit vs Train-Only Fit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")
    
    # Print summary
    same_last = regimes_full[-1] == regimes_train[-1]
    print(f"\n📊 LEAKAGE IMPACT SUMMARY:")
    print(f"  Total days with different regime: {int(diff.sum())} / {len(df)} ({diff.sum()/len(df)*100:.1f}%)")
    print(f"  Last day regime: Full={regimes_full[-1]}, Train-only={regimes_train[-1]}")
    print(f"  Same regime for prediction? {'✓ YES' if same_last else '✗ NO (This explains performance difference!)'}")


def main():
    print("=" * 70)
    print("DEBUG HMM REGIME CLUSTERING")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_and_prepare_data()
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['time'].min().date()} to {df['time'].max().date()}")
    
    # Calculate splits
    train_size = int(len(df) * 0.8)
    print(f"  Train size: {train_size} days")
    print(f"  Val size: {len(df) - train_size} days")
    
    # Fit HMM both ways
    print("\n🔄 Fitting HMM models...")
    regimes_full, model_full, _ = fit_hmm_full(df, N_REGIMES, REGIME_WINDOW)
    regimes_train, model_train, _ = fit_hmm_train_only(df, train_size, N_REGIMES, REGIME_WINDOW)
    print("  ✓ Full-fit HMM done")
    print("  ✓ Train-only HMM done")
    
    # Generate all visualizations
    print("\n📊 Generating visualizations...")
    
    # Plot 1: Regime Timeline
    plot_1_regime_timeline(df, regimes_full, f'{OUTPUT_DIR}/1_regime_timeline.png')
    
    # Plot 2: Data Splitting
    plot_2_regime_data_splitting(df, regimes_full, train_size, SEQ_LEN, 
                                  f'{OUTPUT_DIR}/2_regime_data_splitting.png')
    
    # Plot 3: Hidden Test Mechanism
    current_regime, regime_count = plot_3_hidden_test_prediction_mechanism(
        df, regimes_full, train_size, SEQ_LEN, 
        f'{OUTPUT_DIR}/3_hidden_test_mechanism.png')
    
    # Plot 4: Regime Statistics
    plot_4_regime_statistics(df, regimes_full, f'{OUTPUT_DIR}/4_regime_statistics.png')
    
    # Plot 5: Leakage Comparison
    plot_5_leakage_comparison(df, regimes_full, regimes_train, train_size,
                              f'{OUTPUT_DIR}/5_leakage_comparison.png')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
KEY FINDINGS:
1. Current regime (last day) for prediction: Regime {current_regime}
2. Training samples in this regime: {regime_count}
3. Data leakage impact: See Plot 5

HYPOTHESIS VALIDATION:
- If last-day regime is SAME for both full-fit and train-only:
  → Data leakage may not hurt because prediction uses same regime
- If training data within regime has MANY gaps:
  → Model learns from non-consecutive data (may or may not hurt)

All plots saved to: {OUTPUT_DIR}/
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
