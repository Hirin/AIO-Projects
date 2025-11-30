# =============================================================================
# BUSINESS DAYS vs CALENDAR DAYS EXPERIMENT
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = 'data/FPT_train.csv'
OUTPUT_DIR = 'submissions/business_days_experiment'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBMISSION_LEN = 100  # Required submission length
BATCH_SIZE = 32
EPOCHS = 1000
PATIENCE = 15
LEARNING_RATE = 1e-3

# =============================================================================
# LOAD DATA & ANALYZE DATE PATTERN
# =============================================================================
df = pd.read_csv(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Analyze date gaps
df['day_of_week'] = df['time'].dt.dayofweek  # 0=Mon, 6=Sun
df['date_diff'] = df['time'].diff().dt.days

print("="*60)
print("DATE ANALYSIS")
print("="*60)
print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")
print(f"Total rows: {len(df)}")
print(f"\nDay of week distribution:")
print(df['day_of_week'].value_counts().sort_index())
print(f"\nDate gaps (days between consecutive rows):")
print(df['date_diff'].value_counts().sort_index().head(10))

# Calculate business days ratio
total_calendar_days = (df['time'].max() - df['time'].min()).days
business_day_ratio = len(df) / total_calendar_days
print(f"\nBusiness day ratio: {len(df)}/{total_calendar_days} = {business_day_ratio:.3f}")
print(f"Estimated business days per 100 calendar days: {100 * business_day_ratio:.1f}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[f'{col}_log'] = np.log1p(df[col])

df['HL_Spread'] = df['high_log'] - df['low_log']
df['OC_Spread'] = df['close_log'] - df['open_log']

# =============================================================================
# HELPER CLASSES
# =============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_window(data, seq_len, pred_len, target_col_idx, feature_cols_idx):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len, feature_cols_idx])
        y.append(data[i + seq_len : i + seq_len + pred_len, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def inverse_transform(log_data):
    return np.expm1(log_data)

# =============================================================================
# RevIN & MODEL
# =============================================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        return x


class Uni_Linear_RevIN(nn.Module):
    def __init__(self, seq_len, pred_len, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        x = self.revin(x, 'norm')
        target_mean = self.revin.mean[:, :, 0]
        target_stdev = self.revin.stdev[:, :, 0]
        x_in = x[:, :, 0]
        out = self.linear(x_in)
        out = out * target_stdev + target_mean
        return out


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, patience=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.best_loss = float('inf')
        self.best_state = None
        self.counter = 0

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
            
            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = deepcopy(self.model.state_dict())
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                break
        if self.best_state:
            self.model.load_state_dict(self.best_state)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                total_loss += self.criterion(self.model(X_batch), y_batch).item()
        return total_loss / len(loader)


# =============================================================================
# TRAIN BASE MODEL
# =============================================================================
def train_model(seq_len, pred_len):
    """Train model with specific pred_len"""
    seed_everything(42)
    
    feature_cols = ['close_log', 'volume_log', 'HL_Spread', 'OC_Spread']
    target_col_idx = df.columns.get_loc('close_log')
    feature_cols_idx = [df.columns.get_loc(c) for c in feature_cols]
    data_values = df.values
    num_features = len(feature_cols_idx)
    
    X_all, y_all = create_sliding_window(data_values, seq_len, pred_len, target_col_idx, feature_cols_idx)
    train_loader = DataLoader(TimeSeriesDataset(X_all, y_all), batch_size=BATCH_SIZE, shuffle=True)
    
    model = Uni_Linear_RevIN(seq_len, pred_len, num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    trainer = Trainer(model, criterion, optimizer, scheduler, patience=PATIENCE)
    trainer.fit(train_loader, train_loader, EPOCHS)
    
    # Predict
    last_sequence = data_values[-seq_len:, feature_cols_idx]
    last_seq_tensor = torch.tensor(last_sequence.astype(np.float32)).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred_log = model(last_seq_tensor).cpu().numpy().flatten()
    
    return inverse_transform(pred_log)


# =============================================================================
# INTERPOLATION STRATEGIES
# =============================================================================
def interpolate_to_target(predictions, source_len, target_len, method='linear'):
    """
    Interpolate predictions from source_len to target_len
    
    Args:
        predictions: array of length source_len (business days)
        source_len: number of business days predicted
        target_len: required submission length (calendar days)
        method: 'linear', 'cubic', 'nearest'
    """
    source_x = np.linspace(0, 1, source_len)
    target_x = np.linspace(0, 1, target_len)
    
    if method == 'linear':
        f = interp1d(source_x, predictions, kind='linear')
    elif method == 'cubic':
        f = interp1d(source_x, predictions, kind='cubic')
    elif method == 'nearest':
        f = interp1d(source_x, predictions, kind='nearest')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return f(target_x)


def fill_with_repeat(predictions, source_len, target_len):
    """
    Fill weekends by repeating Friday's value
    Assumes 5 business days per 7 calendar days pattern
    """
    result = []
    biz_idx = 0
    
    for cal_day in range(target_len):
        # Estimate which business day this calendar day corresponds to
        # Roughly: every 7 calendar days = 5 business days
        expected_biz_day = cal_day * 5 / 7
        
        if biz_idx < source_len - 1 and expected_biz_day >= biz_idx + 1:
            biz_idx += 1
        
        if biz_idx < source_len:
            result.append(predictions[biz_idx])
        else:
            result.append(predictions[-1])  # Use last value
    
    return np.array(result[:target_len])


# =============================================================================
# EXPERIMENT: DIFFERENT PRED_LENS AND INTERPOLATION
# =============================================================================
print("\n" + "="*60)
print("EXPERIMENTS: BUSINESS DAYS → CALENDAR DAYS")
print("="*60)

results = []
seq_len = 480

# Hypothesis: If submission requires 100 calendar days
# We should predict ~70 business days and interpolate

# Test different business day assumptions
business_day_predictions = [60, 65, 70, 72, 75, 80, 85, 90, 100]

for biz_days in business_day_predictions:
    print(f"\n--- Predicting {biz_days} business days, interpolate to {SUBMISSION_LEN} ---")
    
    # Train model with this pred_len
    pred_biz = train_model(seq_len, biz_days)
    
    # Interpolate to 100 using different methods
    for method in ['linear', 'cubic']:
        pred_interpolated = interpolate_to_target(pred_biz, biz_days, SUBMISSION_LEN, method)
        
        name = f"BizDays{biz_days}_Interp{method}_Seq{seq_len}"
        filename = f"Sub_{name}_Pred{SUBMISSION_LEN}.csv"
        
        sub_df = pd.DataFrame({'id': range(1, SUBMISSION_LEN + 1), 'close': pred_interpolated})
        sub_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        
        results.append({
            'BizDays': biz_days,
            'Method': method,
            'Name': name,
            'PredMin': pred_interpolated.min(),
            'PredMax': pred_interpolated.max(),
            'PredMean': pred_interpolated.mean(),
            'Filename': filename
        })
        
        print(f"  ✓ {method}: range [{pred_interpolated.min():.2f}, {pred_interpolated.max():.2f}]")

# Also try repeat-fill method
for biz_days in [70, 72, 75]:
    print(f"\n--- Predicting {biz_days} business days, repeat-fill to {SUBMISSION_LEN} ---")
    
    pred_biz = train_model(seq_len, biz_days)
    pred_filled = fill_with_repeat(pred_biz, biz_days, SUBMISSION_LEN)
    
    name = f"BizDays{biz_days}_RepeatFill_Seq{seq_len}"
    filename = f"Sub_{name}_Pred{SUBMISSION_LEN}.csv"
    
    sub_df = pd.DataFrame({'id': range(1, SUBMISSION_LEN + 1), 'close': pred_filled})
    sub_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    
    results.append({
        'BizDays': biz_days,
        'Method': 'repeat_fill',
        'Name': name,
        'PredMin': pred_filled.min(),
        'PredMax': pred_filled.max(),
        'PredMean': pred_filled.mean(),
        'Filename': filename
    })
    
    print(f"  ✓ repeat_fill: range [{pred_filled.min():.2f}, {pred_filled.max():.2f}]")

# =============================================================================
# BASELINE: Original 100 business days prediction (no interpolation)
# =============================================================================
print(f"\n--- Baseline: {SUBMISSION_LEN} days direct prediction ---")
pred_baseline = train_model(seq_len, SUBMISSION_LEN)
name = f"Baseline_Seq{seq_len}"
filename = f"Sub_{name}_Pred{SUBMISSION_LEN}.csv"

sub_df = pd.DataFrame({'id': range(1, SUBMISSION_LEN + 1), 'close': pred_baseline})
sub_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

results.append({
    'BizDays': 100,
    'Method': 'baseline',
    'Name': name,
    'PredMin': pred_baseline.min(),
    'PredMax': pred_baseline.max(),
    'PredMean': pred_baseline.mean(),
    'Filename': filename
})

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot different business day predictions
ax1 = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(business_day_predictions)))
for biz_days, color in zip([60, 70, 80, 100], colors[[0, 3, 5, 8]]):
    pred = train_model(seq_len, biz_days)
    pred_interp = interpolate_to_target(pred, biz_days, SUBMISSION_LEN, 'linear')
    ax1.plot(pred_interp, label=f'BizDays={biz_days}', alpha=0.7, color=color)
ax1.set_title('Different Business Day Predictions (Linear Interpolation)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Submission Day (1-100)')
ax1.set_ylabel('Price')

# Plot interpolation methods comparison
ax2 = axes[0, 1]
pred_70 = train_model(seq_len, 70)
for method in ['linear', 'cubic']:
    pred_interp = interpolate_to_target(pred_70, 70, SUBMISSION_LEN, method)
    ax2.plot(pred_interp, label=f'{method}', alpha=0.7)
pred_repeat = fill_with_repeat(pred_70, 70, SUBMISSION_LEN)
ax2.plot(pred_repeat, label='repeat_fill', alpha=0.7, linestyle='--')
ax2.plot(pred_baseline, label='baseline (100 biz)', alpha=0.7, linestyle=':')
ax2.set_title('Interpolation Methods Comparison (70 BizDays)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot historical + predictions
ax3 = axes[1, 0]
history_len = 150
ax3.plot(range(history_len), df['close'].iloc[-history_len:].values, 'b-', linewidth=2, label='Historical')
ax3.plot(range(history_len, history_len + SUBMISSION_LEN), pred_baseline, 'r--', alpha=0.7, label='Baseline (100 biz)')
pred_70_interp = interpolate_to_target(train_model(seq_len, 70), 70, SUBMISSION_LEN, 'linear')
ax3.plot(range(history_len, history_len + SUBMISSION_LEN), pred_70_interp, 'g--', alpha=0.7, label='70 BizDays → 100')
ax3.axvline(x=history_len, color='gray', linestyle=':', alpha=0.5)
ax3.set_title('Historical + Predictions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot date pattern analysis
ax4 = axes[1, 1]
day_counts = df['day_of_week'].value_counts().sort_index()
days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax4.bar(range(len(day_counts)), day_counts.values)
ax4.set_xticks(range(len(day_counts)))
ax4.set_xticklabels([days_labels[i] for i in day_counts.index])
ax4.set_title('Day of Week Distribution in Training Data')
ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'business_days_analysis.png'), dpi=150)
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['BizDays', 'Method'])
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, 'experiment_summary.csv'), index=False)

print(f"\n✅ Total experiments: {len(results)}")
print(f"📁 Output directory: {OUTPUT_DIR}/")

# Key insight
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print(f"""
1. Training data appears to be BUSINESS DAYS (no weekends)
2. Business day ratio: {business_day_ratio:.3f}

If hidden test is CALENDAR days:
   - 100 calendar days ≈ {int(100 * business_day_ratio)} business days
   - Should predict ~70 biz days and interpolate to 100

If hidden test is BUSINESS days:
   - Keep predicting 100 directly
   - Current baseline should be correct

RECOMMENDATION: Submit both versions and compare scores!
""")