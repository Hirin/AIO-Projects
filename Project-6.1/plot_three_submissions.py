"""
Script vẽ plot 3 submission nối tiếp trên dữ liệu train FPT
- Sub_Multivariate_DLinear_HMM3W60_Seq60_MSE550.csv (Private Score: 14.3493)
- Sub_Multivariate_Linear_HMM3W30_Seq60_MSE1261.csv (Private Score: 21.5984)
- Sub_Univariate_DLinear_NoHMM_Seq480_MSE4118.csv (Private Score: 22.9383)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Thiết lập style cho plot
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Đường dẫn
DATA_DIR = './data'
SUBMISSIONS_DIR = './submissions/FPT'

# 4 file submission cần vẽ (Top 4 Private Score)
# Univariate = Warm colors, Multivariate = Cool colors
SUBMISSION_FILES = [
    # Warm Colors for Univariate
    ('Sub_Univariate_DLinear_NoHMM_Seq480_MSE4118.csv', 'Univ. DLinear Seq480 (Private: 28.98)', '#e74c3c'), # Red
    ('Sub_Univariate_Linear_NoHMM_Seq480_MSE4188.csv', 'Univ. Linear Seq480 (Private: 39.81)', '#d35400'), # Dark Orange
    
    # Cool Colors for Multivariate
    ('Sub_Multivariate_DLinear_HMM3W60_Seq60_MSE550.csv', 'Multi. DLinear HMM (Private: 47.60)', '#2980b9'), # Blue
    ('Sub_Multivariate_Linear_HMM3W60_Seq60_MSE633.csv', 'Multi. Linear HMM (Private: 66.89)', '#1abc9c'), # Teal
]

def load_data():
    """Load dữ liệu train và các submission"""
    # Load train data
    train_df = pd.read_csv(f'{DATA_DIR}/FPT_train.csv')
    train_df['time'] = pd.to_datetime(train_df['time'])
    
    # Load các submission
    submissions = {}
    for filename, label, color in SUBMISSION_FILES:
        try:
            sub_df = pd.read_csv(f'{SUBMISSIONS_DIR}/{filename}')
            submissions[label] = {
                'data': sub_df,
                'color': color,
                'filename': filename
            }
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    return train_df, submissions

def plot_combined(train_df, submissions):
    """Vẽ plot 4 submission combined"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Vẽ dữ liệu train (last 100 days)
    train_subset = train_df.iloc[-100:]
    ax.plot(train_subset['time'], train_subset['close'], label='History (Last 100 days)', color='black', linewidth=1.5, alpha=0.5)
    
    last_date = pd.to_datetime(train_subset['time'].iloc[-1])
    
    for label, info in submissions.items():
        sub_data = info['data']
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(sub_data))
        ax.plot(dates, sub_data['close'], label=label, color=info['color'], linewidth=2, alpha=0.9)
    
    ax.set_title('Combined Forecast Comparison: Univariate (Warm) vs Multivariate (Cool)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = './images/four_models_combined.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

def plot_grid(train_df, submissions):
    """Vẽ plot 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    train_subset = train_df.iloc[-100:]
    last_date = pd.to_datetime(train_subset['time'].iloc[-1])
    
    for idx, (label, info) in enumerate(submissions.items()):
        if idx >= 4: break
        
        ax = axes[idx]
        # Plot History
        ax.plot(train_subset['time'], train_subset['close'], label='History', color='gray', alpha=0.6)
        
        # Plot Forecast
        sub_data = info['data']
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(sub_data))
        ax.plot(dates, sub_data['close'], label=label, color=info['color'], linewidth=2.5)
        
        ax.set_title(label, fontsize=12, fontweight='bold', color=info['color'])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Individual Model Forecasts', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = './images/four_models_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

def main():
    print("Loading data...")
    train_df, submissions = load_data()
    
    print("Plotting Combined...")
    plot_combined(train_df, submissions)
    
    print("Plotting Grid...")
    plot_grid(train_df, submissions)
    
    print("Done!")

if __name__ == "__main__":
    main()
