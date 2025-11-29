# =============================================================================
# PRED_LEN TUNING - Focused Experiments (with HMM Support) - FIXED
# =============================================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from hmmlearn.hmm import GaussianHMM

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
    torch.backends.cudnn.benchmark = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = 'data/FPT_train.csv'
OUTPUT_DIR = 'submissions/pred_len_tuning'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_LENS = [100, 120, 150, 200, 220]

BEST_CONFIGS = [
    {'variant': 'Univariate', 'model': 'Linear', 'use_revin': True, 'seq_len': 480, 'use_hmm': False, 'n_regimes': None, 'regime_window': None},
    {'variant': 'Univariate', 'model': 'DLinear', 'use_revin': True, 'seq_len': 480, 'use_hmm': False, 'n_regimes': None, 'regime_window': None},
    {'variant': 'Multivariate', 'model': 'DLinear', 'use_revin': True, 'seq_len': 60, 'use_hmm': True, 'n_regimes': 3, 'regime_window': 60},
    {'variant': 'Multivariate', 'model': 'Linear', 'use_revin': True, 'seq_len': 60, 'use_hmm': True, 'n_regimes': 3, 'regime_window': 60},
    {'variant': 'Multivariate', 'model': 'Linear', 'use_revin': True, 'seq_len': 60, 'use_hmm': True, 'n_regimes': 3, 'regime_window': 30},
    {'variant': 'Multivariate', 'model': 'DLinear', 'use_revin': True, 'seq_len': 60, 'use_hmm': True, 'n_regimes': 3, 'regime_window': 30},
]

BATCH_SIZE = 32
EPOCHS = 1000
PATIENCE = 15
LEARNING_RATE = 1e-3

print(f"PRED_LENS to test: {PRED_LENS}")
print(f"Models to test: {len(BEST_CONFIGS)}")
print(f"Total experiments: {len(PRED_LENS) * len(BEST_CONFIGS)}")

# =============================================================================
# LOAD DATA & FEATURE ENGINEERING
# =============================================================================
df = pd.read_csv(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

for col in ['open', 'high', 'low', 'close', 'volume']:
    df[f'{col}_log'] = np.log1p(df[col])
df['HL_Spread'] = df['high_log'] - df['low_log']
df['OC_Spread'] = df['close_log'] - df['open_log']
df['returns'] = df['close'].pct_change().fillna(0)
df['volatility'] = df['returns'].rolling(window=10).std().fillna(0)
df['trend'] = df['close'].rolling(window=10).mean().pct_change().fillna(0)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")

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
# REGIME DETECTOR - COPY EXACT FROM ORIGINAL
# =============================================================================
class RegimeDetector:
    def __init__(self, n_components=3, window=30):
        self.n_components = n_components
        self.window = window
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

    def fit_predict(self, df):
        features = df[['returns', 'volatility', 'trend']].iloc[self.window:].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.model.fit(features_scaled)
        states = self.model.predict(features_scaled)
        full_states = np.concatenate([np.zeros(self.window) - 1, states])
        return full_states

# =============================================================================
# RevIN - COPY EXACT FROM ORIGINAL (QUAN TRỌNG!)
# =============================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization - EXACT COPY from original notebook
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

# =============================================================================
# MODELS - COPY EXACT FROM ORIGINAL
# =============================================================================
class Uni_Linear_RevIN(nn.Module):
    """Univariate Linear with RevIN"""
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


class Uni_DLinear_RevIN(nn.Module):
    """Univariate DLinear with RevIN"""
    def __init__(self, seq_len, pred_len, num_features):
        super().__init__()
        self.seq_len = seq_len
        self.revin = RevIN(num_features)
        kernel_size = min(25, seq_len)
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        target_mean = self.revin.mean[:, :, 0]
        target_stdev = self.revin.stdev[:, :, 0]
        
        x_in = x[:, :, 0].unsqueeze(1)  # [B, 1, S]
        trend = self.moving_avg(x_in).squeeze(1)
        if trend.shape[1] != self.seq_len:
            trend = trend[:, :self.seq_len]
        seasonal = x[:, :, 0] - trend
        
        out = self.linear_trend(trend) + self.linear_seasonal(seasonal)
        out = out * target_stdev + target_mean
        return out


class Multi_Linear_RevIN(nn.Module):
    """Multivariate Linear with RevIN"""
    def __init__(self, seq_len, pred_len, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.linear = nn.Linear(seq_len * num_features, pred_len)
        
    def forward(self, x):
        x = self.revin(x, 'norm')
        target_mean = self.revin.mean[:, :, 0]
        target_stdev = self.revin.stdev[:, :, 0]
        
        x_flat = x.reshape(x.shape[0], -1)
        out = self.linear(x_flat)
        out = out * target_stdev + target_mean
        return out


class Multi_DLinear_RevIN(nn.Module):
    """Multivariate DLinear with RevIN"""
    def __init__(self, seq_len, pred_len, num_features):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.revin = RevIN(num_features)
        
        kernel_size = min(25, seq_len)
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.linear_trend = nn.Linear(seq_len * num_features, pred_len)
        self.linear_seasonal = nn.Linear(seq_len * num_features, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        target_mean = self.revin.mean[:, :, 0]
        target_stdev = self.revin.stdev[:, :, 0]
        
        # [B, S, F] -> [B, F, S] for AvgPool
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        if trend.shape[1] != self.seq_len:
            trend = trend[:, :self.seq_len, :]
        seasonal = x - trend
        
        trend_flat = trend.reshape(trend.shape[0], -1)
        seasonal_flat = seasonal.reshape(seasonal.shape[0], -1)
        
        out = self.linear_trend(trend_flat) + self.linear_seasonal(seasonal_flat)
        out = out * target_stdev + target_mean
        return out


MODEL_REGISTRY = {
    ('Univariate', 'Linear', True): Uni_Linear_RevIN,
    ('Univariate', 'DLinear', True): Uni_DLinear_RevIN,
    ('Multivariate', 'Linear', True): Multi_Linear_RevIN,
    ('Multivariate', 'DLinear', True): Multi_DLinear_RevIN,
}

def create_model(variant, model_type, use_revin, seq_len, pred_len, num_features):
    return MODEL_REGISTRY[(variant, model_type, use_revin)](seq_len, pred_len, num_features)

# =============================================================================
# TRAINER - COPY EXACT FROM ORIGINAL
# =============================================================================
class Trainer:
    """Training manager with early stopping and learning rate scheduling."""
    def __init__(self, model, criterion, optimizer, scheduler, patience=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.best_loss = float('inf')
        self.best_state = None
        self.counter = 0
        
    def fit(self, train_loader, val_loader, epochs, verbose=False):
        """Train the model with early stopping"""
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
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
        """Evaluate model on a dataset"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

# =============================================================================
# TRAINING & EVALUATION - COPY EXACT LOGIC FROM ORIGINAL
# =============================================================================
def train_model_func(variant, model_type, use_revin, seq_len, pred_len, num_features, X_train, y_train, epochs=EPOCHS):
    """Helper function to train a single model - MATCHES ORIGINAL LOGIC"""
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    model = create_model(variant, model_type, use_revin, seq_len, pred_len, num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    trainer = Trainer(model, criterion, optimizer, scheduler, patience=PATIENCE)
    trainer.fit(train_loader, train_loader, epochs)
    
    return model


def evaluate_model_func(model, X, y):
    """Evaluate model and compute MSE on original price scale - MATCHES ORIGINAL"""
    model.eval()
    loader = DataLoader(TimeSeriesDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    
    preds_log, trues_log = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            preds_log.append(out.cpu().numpy())
            trues_log.append(y_b.numpy())
    
    preds_log = np.concatenate(preds_log)
    trues_log = np.concatenate(trues_log)
    
    preds_price = inverse_transform(preds_log)
    trues_price = inverse_transform(trues_log)
    
    return mean_squared_error(trues_price.flatten(), preds_price.flatten())


def evaluate_with_hmm(global_model, variant, model_type, use_revin, seq_len, pred_len, num_features,
                      X_train, y_train, X_val, y_val, regimes, train_size):
    """Evaluate with regime-switching models - COPY EXACT FROM ORIGINAL"""
    regime_models = {}
    train_regimes = regimes[:train_size]
    unique_regimes = np.unique(train_regimes)
    
    # Map index X_train sang Regime
    train_regime_indices = []
    for i in range(len(X_train)):
        r_idx = i + seq_len - 1
        if r_idx < len(train_regimes):
            train_regime_indices.append(train_regimes[r_idx])
        else:
            train_regime_indices.append(-1)
    train_regime_indices = np.array(train_regime_indices)
    
    # Train model riêng cho từng regime
    for r in unique_regimes:
        if r == -1:
            continue
        mask = (train_regime_indices == r)
        if mask.sum() > 30:
            X_r = X_train[mask]
            y_r = y_train[mask]
            regime_models[r] = train_model_func(variant, model_type, use_revin, seq_len, pred_len, num_features, X_r, y_r)
    
    val_preds_log, val_trues_log = [], []
    
    global_model.eval()
    for model in regime_models.values():
        model.eval()
    
    with torch.no_grad():
        for i in range(len(X_val)):
            # Xác định regime của mẫu validation hiện tại
            global_idx = train_size + i - 1
            curr_regime = regimes[global_idx] if global_idx < len(regimes) else -1
            
            # Chọn model: Nếu có model regime thì dùng, ko thì dùng global
            selected_model = regime_models.get(curr_regime, global_model)
            
            inp = torch.tensor(X_val[i]).unsqueeze(0).to(device)
            pred = selected_model(inp).cpu().numpy()
            
            val_preds_log.append(pred)
            val_trues_log.append(y_val[i])
    
    val_preds_log = np.concatenate(val_preds_log)
    val_trues_log = np.array(val_trues_log)
    
    pred_price = inverse_transform(val_preds_log)
    true_price = inverse_transform(val_trues_log)
    
    return mean_squared_error(true_price.flatten(), pred_price.flatten())

# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
print("\n" + "="*70)
print("STARTING PRED_LEN TUNING EXPERIMENTS (FIXED VERSION)")
print("="*70)

results = []
feature_cols = ['close_log', 'volume_log', 'HL_Spread', 'OC_Spread']
target_col_idx = df.columns.get_loc('close_log')
feature_cols_idx = [df.columns.get_loc(c) for c in feature_cols]
data_values = df.values

total_experiments = len(PRED_LENS) * len(BEST_CONFIGS)
pbar = tqdm(total=total_experiments, desc="Experiments")

for pred_len in PRED_LENS:
    print(f"\n{'='*50}")
    print(f"PRED_LEN = {pred_len}")
    print(f"{'='*50}")
    
    for config in BEST_CONFIGS:
        seed_everything(42)
        
        variant = config['variant']
        model_type = config['model']
        use_revin = config['use_revin']
        seq_len = config['seq_len']
        use_hmm = config['use_hmm']
        n_regimes = config['n_regimes']
        regime_window = config['regime_window']
        
        num_features = len(feature_cols_idx)
        
        # HMM Regime Detection
        regimes = None
        if use_hmm:
            detector = RegimeDetector(n_components=n_regimes, window=regime_window)
            regimes = detector.fit_predict(df)
        
        # Train/Val split (80/20)
        train_size = int(len(data_values) * 0.8)
        train_data = data_values[:train_size]
        val_data = data_values[train_size - seq_len:]
        
        # Create sliding windows
        X_train, y_train = create_sliding_window(train_data, seq_len, pred_len, target_col_idx, feature_cols_idx)
        X_val, y_val = create_sliding_window(val_data, seq_len, pred_len, target_col_idx, feature_cols_idx)
        
        # Skip if not enough data
        if len(X_train) < 10 or len(X_val) < 1:
            print(f"  ⚠ Skipping {variant}_{model_type}_Seq{seq_len} - Not enough data for pred_len={pred_len}")
            pbar.update(1)
            continue
        
        # Train global model
        global_model = train_model_func(variant, model_type, use_revin, seq_len, pred_len, num_features, X_train, y_train)
        
        # Evaluate
        if not use_hmm:
            val_mse = evaluate_model_func(global_model, X_val, y_val)
        else:
            val_mse = evaluate_with_hmm(
                global_model, variant, model_type, use_revin,
                seq_len, pred_len, num_features,
                X_train, y_train, X_val, y_val,
                regimes, train_size
            )
        
        # Production forecast (train on 95% data)
        prod_train_size = int(len(data_values) * 0.95)
        prod_train_data = data_values[:prod_train_size]
        X_prod, y_prod = create_sliding_window(prod_train_data, seq_len, pred_len, target_col_idx, feature_cols_idx)
        
        final_model = train_model_func(variant, model_type, use_revin, seq_len, pred_len, num_features, X_prod, y_prod, epochs=EPOCHS//2)
        
        # Generate predictions
        last_sequence = data_values[-seq_len:, feature_cols_idx]
        last_seq_tensor = torch.tensor(last_sequence.astype(np.float32)).unsqueeze(0).to(device)
        
        final_model.eval()
        with torch.no_grad():
            pred_log = final_model(last_seq_tensor).cpu().numpy().flatten()
        
        # HMM Regime correction for production
        if use_hmm:
            current_regime = regimes[-1] if len(regimes) > 0 else -1
            
            prod_regime_indices = []
            for i in range(len(X_prod)):
                r_idx = i + seq_len - 1
                if r_idx < len(regimes):
                    prod_regime_indices.append(regimes[r_idx])
                else:
                    prod_regime_indices.append(-1)
            
            prod_regime_indices = np.array(prod_regime_indices)
            mask = (prod_regime_indices == current_regime)
            
            if mask.sum() > 30:
                X_regime = X_prod[mask]
                y_regime = y_prod[mask]
                regime_model = train_model_func(variant, model_type, use_revin, seq_len, pred_len, num_features, 
                                               X_regime, y_regime, epochs=EPOCHS//2)
                regime_model.eval()
                with torch.no_grad():
                    pred_log = regime_model(last_seq_tensor).cpu().numpy().flatten()
        
        pred_price = inverse_transform(pred_log)
        
        # Build filename - MATCH ORIGINAL FORMAT
        revin_str = "RevIN" if use_revin else "NoRevIN"
        hmm_str = f"HMM{n_regimes}W{regime_window}" if use_hmm else "NoHMM"
        filename = f"Sub_{variant}_{model_type}_{revin_str}_MSE_{hmm_str}_Seq{seq_len}_Pred{pred_len}_ValMSE{val_mse:.0f}.csv"
        
        sub_df = pd.DataFrame({'id': range(1, pred_len + 1), 'close': pred_price})
        filepath = os.path.join(OUTPUT_DIR, filename)
        sub_df.to_csv(filepath, index=False)
        
        results.append({
            'Variant': variant,
            'Model': model_type,
            'RevIN': revin_str,
            'HMM': hmm_str,
            'SeqLen': seq_len,
            'PredLen': pred_len,
            'ValMSE': val_mse,
            'File': filename
        })
        
        print(f"  ✓ {variant}_{model_type}_{hmm_str}_Seq{seq_len} | ValMSE={val_mse:.2f}")
        pbar.update(1)

pbar.close()

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['PredLen', 'ValMSE'])
results_df.to_csv(os.path.join(OUTPUT_DIR, 'pred_len_tuning_results.csv'), index=False)

print(f"\nTotal experiments: {len(results_df)}")

# Display results grouped by PredLen
for pred_len in PRED_LENS:
    print(f"\n📊 PRED_LEN = {pred_len}:")
    subset = results_df[results_df['PredLen'] == pred_len].sort_values('ValMSE')
    for _, row in subset.iterrows():
        print(f"   {row['Variant']}_{row['Model']}_{row['HMM']}_Seq{row['SeqLen']} | ValMSE={row['ValMSE']:.2f}")

# Pivot table
print("\n" + "-"*70)
print("\n📈 ValMSE Comparison Table:")
pivot = results_df.pivot_table(
    values='ValMSE', 
    index=['Variant', 'Model', 'HMM', 'SeqLen'], 
    columns='PredLen', 
    aggfunc='first'
)
print(pivot.round(2))

print(f"\n✅ All files saved to: {OUTPUT_DIR}/")