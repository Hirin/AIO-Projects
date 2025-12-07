"""
Script vẽ các biểu đồ so sánh chi tiết cho Blog
1. SeqLen Impact: Univariate Seq480 (Best) vs Seq60 (Overfit)
2. HMM Impact: Multivariate HMM (Stable) vs NoHMM (Volatile)
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

DATA_DIR = './data'
SUBMISSIONS_DIR = './submissions/FPT'

def load_data(filename):
    try:
        return pd.read_csv(f'{SUBMISSIONS_DIR}/{filename}')
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

def plot_comparison(train_df, file1, file2, title, output_filename):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot last 100 days of train for context
    train_subset = train_df.iloc[-100:]
    ax.plot(train_subset['time'], train_subset['close'], label='History (Last 100 days)', color='black', alpha=0.5)
    
    # Load and plot sub 1
    df1 = load_data(file1[0])
    if df1 is not None:
        # Create time index for forecast (assuming daily after last train date)
        last_date = pd.to_datetime(train_subset['time'].iloc[-1])
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df1))
        ax.plot(dates, df1['close'], label=f"{file1[1]}", color=file1[2], linewidth=2)
        
    # Load and plot sub 2
    df2 = load_data(file2[0])
    if df2 is not None:
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df2))
        ax.plot(dates, df2['close'], label=f"{file2[1]}", color=file2[2], linewidth=2, linestyle='--')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'./images/{output_filename}', bbox_inches='tight')
    print(f"Saved {output_filename}")
    plt.close()

def main():
    train_df = pd.read_csv(f'{DATA_DIR}/FPT_train.csv')
    train_df['time'] = pd.to_datetime(train_df['time'])
    
    # 1. SeqLen Analysis (Warm Colors)
    # Seq480 (Private: 28.98) vs Seq60 (Private: ~203 - Overfit)
    plot_comparison(
        train_df,
        ('Sub_Univariate_DLinear_NoHMM_Seq480_MSE4118.csv', 'Seq480 (Private: 28.98)', '#e74c3c'), # Red (Best)
        ('Sub_Univariate_DLinear_NoHMM_Seq60_MSE190.csv', 'Seq60 (Private: ~203)', '#e67e22'), # Orange (Overfit)
        'Impact of Sequence Length: Long-term (Seq480) vs Short-term (Seq60)',
        'analysis_seqlen.png'
    )
    
    # 2. HMM Analysis (Cool Colors)
    # HMM (Private: 47.60) vs NoHMM (Private: ~249)
    plot_comparison(
        train_df,
        ('Sub_Multivariate_DLinear_HMM3W60_Seq60_MSE550.csv', 'With HMM (Private: 47.60)', '#2980b9'), # Blue (Best)
        ('Sub_Multivariate_DLinear_NoHMM_Seq60_MSE193.csv', 'Without HMM (Private: ~249)', '#7f8c8d'), # Grey (Baseline)
        'Impact of HMM Regime Detection on Multivariate Models',
        'analysis_hmm.png'
    )

if __name__ == "__main__":
    main()
