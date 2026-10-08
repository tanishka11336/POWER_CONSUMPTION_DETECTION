import pandas as pd
import numpy as np
import json
import os

data_path = "power consumption.csv"
print(f"Loading {data_path}...")
df = pd.read_csv(data_path)

# Clean column names
df.columns = [c.strip() for c in df.columns]
print("Columns:", df.columns.tolist())
print(f"Shape: {df.shape}")

print("\nMissing values:")
print(df.isnull().sum())

# Date parsing
df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
print(f"Start date: {df['DateTime'].min()}, End date: {df['DateTime'].max()}")

# Total power consumption
df['Total Power'] = df['Zone 1'] + df['Zone 2'] + df['Zone 3']

numeric_cols = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows', 'Zone 1', 'Zone 2', 'Zone 3', 'Total Power']
stats = df[numeric_cols].describe().to_dict()
corr = df[numeric_cols].corr().to_dict()

# Hourly stats
df['Hour'] = df['DateTime'].dt.hour
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

hourly_load = df.groupby('Hour')[['Zone 1', 'Zone 2', 'Zone 3', 'Total Power']].mean().to_dict()
monthly_load = df.groupby('Month')[['Zone 1', 'Zone 2', 'Zone 3', 'Total Power']].mean().to_dict()

os.makedirs("data_processed", exist_ok=True)
summary = {
    "num_rows": len(df),
    "date_range": [str(df['DateTime'].min()), str(df['DateTime'].max())],
    "columns": df.columns.tolist(),
    "summary_stats": {k: {stat: round(val, 2) for stat, val in v.items()} for k, v in stats.items()},
    "correlations": {k: {k2: round(v2, 4) for k2, v2 in v.items()} for k, v in corr.items()},
    "hourly_averages": {k: {h: round(val, 2) for h, val in v.items()} for k, v in hourly_load.items()},
    "monthly_averages": {k: {m: round(val, 2) for m, val in v.items()} for k, v in monthly_load.items()}
}

with open("data_processed/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nEDA completed successfully. Summary saved to data_processed/eda_summary.json")
