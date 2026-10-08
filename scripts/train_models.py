import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print(f"Training device: {device}")

# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("power consumption.csv")
df.columns = [c.strip() for c in df.columns]

# Parse DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
df = df.sort_values('DateTime').reset_index(drop=True)

# Feature Engineering
df['Hour'] = df['DateTime'].dt.hour
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Cyclical features
df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12.0)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12.0)
df['sin_dow'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
df['cos_dow'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

# Physical interaction features
df['HeatIndexProxy'] = (df['Temperature'] * df['Humidity']) / 100.0
df['TotalSolarFlux'] = df['general diffuse flows'] + df['diffuse flows']

# Total Power
df['Total Power'] = df['Zone 1'] + df['Zone 2'] + df['Zone 3']

feature_cols = [
    'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows',
    'HeatIndexProxy', 'TotalSolarFlux',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'IsWeekend',
    'Zone 1', 'Zone 2', 'Zone 3'
]

target_cols = ['Zone 1', 'Zone 2', 'Zone 3']

print(f"Total rows: {len(df)}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Save processed dataframe head/sample for backend quick access
df.iloc[:2000].to_csv("data_processed/sample_stream.csv", index=False)

# Train / Val / Test Chronological Split (70% / 15% / 15%)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

# Scaling - Fit ONLY on Train
scaler_features = StandardScaler()
scaler_targets = StandardScaler()

train_feat_scaled = scaler_features.fit_transform(train_df[feature_cols].values)
val_feat_scaled = scaler_features.transform(val_df[feature_cols].values)
test_feat_scaled = scaler_features.transform(test_df[feature_cols].values)

train_tgt_scaled = scaler_targets.fit_transform(train_df[target_cols].values)
val_tgt_scaled = scaler_targets.transform(val_df[target_cols].values)
test_tgt_scaled = scaler_targets.transform(test_df[target_cols].values)

# Save Scalers parameters
scaler_info = {
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "feature_mean": scaler_features.mean_.tolist(),
    "feature_scale": scaler_features.scale_.tolist(),
    "target_mean": scaler_targets.mean_.tolist(),
    "target_scale": scaler_targets.scale_.tolist(),
    "window_size": 12
}
with open("models/scalers.json", "w") as f:
    json.dump(scaler_info, f, indent=2)

# Sequence Dataset
WINDOW_SIZE = 12  # Past 2 hours (12 steps of 10-min data)

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, features, targets, window_size=12, stride=1):
        self.X = []
        self.y = []
        for i in range(0, len(features) - window_size, stride):
            self.X.append(features[i : i + window_size])
            self.y.append(targets[i + window_size])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print(f"Creating sliding window datasets (W={WINDOW_SIZE})...")
# Use stride=2 for training to speed up while keeping excellent temporal diversity
train_dataset = TimeSeriesWindowDataset(train_feat_scaled, train_tgt_scaled, window_size=WINDOW_SIZE, stride=2)
val_dataset = TimeSeriesWindowDataset(val_feat_scaled, val_tgt_scaled, window_size=WINDOW_SIZE, stride=2)
test_dataset = TimeSeriesWindowDataset(test_feat_scaled, test_tgt_scaled, window_size=WINDOW_SIZE, stride=1)

print(f"Window datasets created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Model Definitions
num_features = len(feature_cols)
num_targets = len(target_cols)

class LSTMRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_features=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRURegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_features=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, in_features, cnn_out=64, lstm_hidden=32, out_features=3, dropout=0.2):
        super().__init__()
        # Conv1d operates on (B, C, L)
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=cnn_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        # x is (B, L, C) -> transpose to (B, C, L)
        x_trans = x.transpose(1, 2)
        conv_feat = self.relu(self.bn1(self.conv1(x_trans)))
        conv_feat = conv_feat.transpose(1, 2) # back to (B, L, C)
        out, _ = self.bilstm(conv_feat)
        return self.fc(out[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_features, d_model=64, nhead=4, num_layers=2, out_features=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        trans_out = self.transformer(x)
        pooled = torch.mean(trans_out, dim=1) # Mean pooling across time
        return self.fc(pooled)

class ResidualMLPRegressor(nn.Module):
    def __init__(self, in_features, window_size=12, hidden_dim=128, out_features=3, dropout=0.15):
        super().__init__()
        self.flatten_dim = in_features * window_size
        self.in_proj = nn.Linear(self.flatten_dim, hidden_dim)
        
        # ResBlock 1
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        # ResBlock 2
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        
        self.out_head = nn.Linear(hidden_dim, out_features)
        
    def forward(self, x):
        b = x.shape[0]
        h = self.in_proj(x.view(b, -1))
        # ResBlock 1
        res1 = h
        h = self.drop1(self.act1(self.fc1(self.ln1(h)))) + res1
        # ResBlock 2
        res2 = h
        h = self.drop2(self.act2(self.fc2(self.ln2(h)))) + res2
        return self.out_head(h)

models_to_train = {
    "LSTM": LSTMRegressor(num_features, hidden_dim=64, num_layers=2, out_features=num_targets),
    "GRU": GRURegressor(num_features, hidden_dim=64, num_layers=2, out_features=num_targets),
    "CNN-BiLSTM": CNNBiLSTMRegressor(num_features, cnn_out=64, lstm_hidden=32, out_features=num_targets),
    "Transformer": TransformerRegressor(num_features, d_model=64, nhead=4, num_layers=2, out_features=num_targets),
    "Residual-MLP": ResidualMLPRegressor(num_features, window_size=WINDOW_SIZE, hidden_dim=128, out_features=num_targets)
}

criterion = nn.HuberLoss(delta=1.0) # Robust to occasional voltage/sensor anomalies
EPOCHS = 12

def train_and_evaluate_model(name, model):
    print(f"\n==================== Training {name} ====================")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = f"models/{name.lower().replace('-', '_')}_best.pt"
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vpred = model(vx)
                vloss = criterion(vpred, vy)
                val_batch_losses.append(vloss.item())
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Huber Loss: {train_loss:.5f} | Val Huber Loss: {val_loss:.5f}")
        
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s. Best Val Loss: {best_val_loss:.5f}")
    
    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Inference Latency Benchmark
    latency_times = []
    test_preds = []
    test_actuals = []
    
    with torch.no_grad():
        for tx, ty in test_loader:
            t0 = time.time()
            tpred = model(tx)
            t1 = time.time()
            latency_times.append((t1 - t0) / len(tx))
            test_preds.append(tpred.numpy())
            test_actuals.append(ty.numpy())
            
    avg_latency_ms = np.mean(latency_times) * 1000.0 # ms per sample
    
    test_preds = np.vstack(test_preds)
    test_actuals = np.vstack(test_actuals)
    
    # Inverse transform to original KW scale
    preds_kw = scaler_targets.inverse_transform(test_preds)
    actuals_kw = scaler_targets.inverse_transform(test_actuals)
    
    # Compute zone-specific & total power metrics
    metrics = {
        "model_name": name,
        "parameters": param_count,
        "training_time_sec": round(train_time, 2),
        "latency_ms_per_sample": round(avg_latency_ms, 3),
        "train_losses": [round(x, 5) for x in train_losses],
        "val_losses": [round(x, 5) for x in val_losses],
        "zones": {}
    }
    
    total_pred = preds_kw.sum(axis=1)
    total_act = actuals_kw.sum(axis=1)
    
    for i, z_name in enumerate(["Zone 1", "Zone 2", "Zone 3"]):
        mae = mean_absolute_error(actuals_kw[:, i], preds_kw[:, i])
        rmse = np.sqrt(mean_squared_error(actuals_kw[:, i], preds_kw[:, i]))
        r2 = r2_score(actuals_kw[:, i], preds_kw[:, i])
        mape = np.mean(np.abs((actuals_kw[:, i] - preds_kw[:, i]) / (actuals_kw[:, i] + 1e-5))) * 100
        metrics["zones"][z_name] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape), 2)
        }
        
    tot_mae = mean_absolute_error(total_act, total_pred)
    tot_rmse = np.sqrt(mean_squared_error(total_act, total_pred))
    tot_r2 = r2_score(total_act, total_pred)
    tot_mape = np.mean(np.abs((total_act - total_pred) / (total_act + 1e-5))) * 100
    
    metrics["Total Power"] = {
        "MAE": round(float(tot_mae), 2),
        "RMSE": round(float(tot_rmse), 2),
        "R2": round(float(tot_r2), 4),
        "MAPE": round(float(tot_mape), 2)
    }
    
    print(f"Test Results for {name}:")
    print(f"Total Power -> MAE: {tot_mae:.2f} KW | RMSE: {tot_rmse:.2f} KW | R2: {tot_r2:.4f} | MAPE: {tot_mape:.2f}% | Latency: {avg_latency_ms:.3f} ms")
    
    # Save a slice of test predictions for UI comparison
    predictions_slice = {
        "actual": actuals_kw[:288].tolist(), # 48 hours of 10-min records
        "predicted": preds_kw[:288].tolist()
    }
    with open(f"data_processed/{name.lower().replace('-', '_')}_sample_preds.json", "w") as f:
        json.dump(predictions_slice, f)
        
    return metrics, (actuals_kw, preds_kw)

# Run benchmark across all models
benchmark_results = {}
all_predictions = {}

for name, model in models_to_train.items():
    metrics, preds = train_and_evaluate_model(name, model)
    benchmark_results[name] = metrics
    all_predictions[name] = preds

with open("data_processed/model_benchmarks.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\nModel benchmarks successfully saved to data_processed/model_benchmarks.json")

# Generate High Quality Visualizations for Report
print("Generating publication-ready figures for technical report...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Loss curves comparison
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for (name, met), col in zip(benchmark_results.items(), colors):
    plt.plot(met['val_losses'], label=f"{name} (Val Loss)", color=col, linewidth=2)
plt.title("Deep Learning Models Validation Convergence (Huber Loss)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Huber Loss", fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("reports/figures/loss_curves.png", dpi=300)
plt.close()

# 2. Actual vs Predicted 48-Hour Trajectory for Best Model (CNN-BiLSTM or LSTM)
best_model_name = min(benchmark_results.keys(), key=lambda k: benchmark_results[k]["Total Power"]["RMSE"])
actuals, preds = all_predictions[best_model_name]

plt.figure(figsize=(14, 6))
time_steps = np.arange(144) * 10 / 60.0 # 24 hours in hours
plt.plot(time_steps, actuals[:144, 0] + actuals[:144, 1] + actuals[:144, 2], label="Actual Total Grid Load (KW)", color="#111827", linewidth=2.5)
plt.plot(time_steps, preds[:144, 0] + preds[:144, 1] + preds[:144, 2], label=f"{best_model_name} Predicted Load (KW)", color="#2563eb", linestyle="--", linewidth=2)
plt.fill_between(time_steps, (preds[:144].sum(axis=1)) * 0.96, (preds[:144].sum(axis=1)) * 1.04, color="#3b82f6", alpha=0.15, label="95% Confidence Band")
plt.title(f"Tetouan Smart Grid - Actual vs Predicted Load Profile (24-Hour Horizon | {best_model_name})", fontsize=14, fontweight='bold')
plt.xlabel("Time Horizon (Hours)", fontsize=12)
plt.ylabel("Power Consumption (KW)", fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png", dpi=300)
plt.close()

# 3. Model Benchmark Bar Chart
plt.figure(figsize=(12, 5))
m_names = list(benchmark_results.keys())
r2_scores = [benchmark_results[m]["Total Power"]["R2"] for m in m_names]
rmse_scores = [benchmark_results[m]["Total Power"]["RMSE"] for m in m_names]

fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Load R² Score', color=color, fontsize=12)
bars = ax1.bar(m_names, r2_scores, color=color, alpha=0.7, width=0.4, align='center')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([min(r2_scores) - 0.05, 1.0])

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('RMSE (KW)', color=color, fontsize=12)
ax2.plot(m_names, rmse_scores, color=color, marker='o', linewidth=2.5, markersize=8)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Comparative Performance: Total Power Forecast R² vs RMSE across Architectures", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/model_comparison_bar.png", dpi=300)
plt.close()

print("All figures saved to reports/figures/ successfully!")
