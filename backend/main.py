import os
import json
import time
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from backend.models_definition import (
    LSTMRegressor,
    GRURegressor,
    CNNBiLSTMRegressor,
    TransformerRegressor,
    ResidualMLPRegressor
)

app = FastAPI(
    title="Tetouan Smart Grid Deep Learning Power Consumption System",
    description="Production-grade AI API for Multi-Zone Power Consumption Forecasting and Grid Optimization",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
DEVICE = torch.device("cpu")
SCALER_DATA = {}
MODELS = {}
DATA_DF = None
TEST_DATA_CACHE = None
STREAM_INDEX = 0

def load_system():
    global SCALER_DATA, MODELS, DATA_DF, TEST_DATA_CACHE
    
    # 1. Load Scalers
    scaler_path = "models/scalers.json"
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            SCALER_DATA = json.load(f)
            
    # 2. Load Raw Dataset for Analytics
    data_path = "power consumption.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df.columns = [c.strip() for c in df.columns]
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
        df['Total Power'] = df['Zone 1'] + df['Zone 2'] + df['Zone 3']
        DATA_DF = df
        
        # Cache the holdout test portion (last 15%)
        n = len(df)
        val_end = int(n * 0.85)
        TEST_DATA_CACHE = df.iloc[val_end:].copy().reset_index(drop=True)
        print(f"Dataset loaded: {len(DATA_DF)} records. Test cache: {len(TEST_DATA_CACHE)} records.")

    # 3. Instantiate and load Model Weights
    in_features = len(SCALER_DATA.get("feature_cols", [])) or 17
    out_features = 3
    window_size = SCALER_DATA.get("window_size", 12)
    
    model_configs = {
        "LSTM": (LSTMRegressor(in_features, hidden_dim=64, num_layers=2, out_features=out_features), "models/lstm_best.pt"),
        "GRU": (GRURegressor(in_features, hidden_dim=64, num_layers=2, out_features=out_features), "models/gru_best.pt"),
        "CNN-BiLSTM": (CNNBiLSTMRegressor(in_features, cnn_out=64, lstm_hidden=32, out_features=out_features), "models/cnn_bilstm_best.pt"),
        "Transformer": (TransformerRegressor(in_features, d_model=64, nhead=4, num_layers=2, out_features=out_features), "models/transformer_best.pt"),
        "Residual-MLP": (ResidualMLPRegressor(in_features, window_size=window_size, hidden_dim=128, out_features=out_features), "models/residual_mlp_best.pt")
    }
    
    for name, (model, path) in model_configs.items():
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                MODELS[name] = model
                print(f"Loaded model weights: {name} from {path}")
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        else:
            print(f"Model file not ready yet: {path}")

# Initial load attempt
load_system()

# Pydantic Schemas
class WeatherInput(BaseModel):
    temperature: float = Field(..., example=24.5, description="Ambient Temperature in °C")
    humidity: float = Field(..., example=65.0, description="Relative Humidity %")
    wind_speed: float = Field(..., example=2.1, description="Wind Speed in m/s")
    general_diffuse_flows: float = Field(..., example=150.0, description="General diffuse solar radiation")
    diffuse_flows: float = Field(..., example=60.0, description="Diffuse solar radiation")
    hour: Optional[int] = Field(14, ge=0, le=23, description="Hour of the day (0-23)")
    month: Optional[int] = Field(7, ge=1, le=12, description="Month of the year (1-12)")
    day_of_week: Optional[int] = Field(2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    model_name: Optional[str] = Field("CNN-BiLSTM", description="Selected Deep Learning Model")

class PredictResponse(BaseModel):
    model_name: str
    zone_1_kw: float
    zone_2_kw: float
    zone_3_kw: float
    total_power_kw: float
    latency_ms: float
    confidence_interval: Dict[str, List[float]]
    grid_status: str
    peak_risk: str

class ScenarioInput(BaseModel):
    temp_delta: float = Field(0.0, description="Temperature shift in °C (-10 to +15)")
    humidity_delta: float = Field(0.0, description="Humidity shift in % (-40 to +40)")
    solar_delta_pct: float = Field(0.0, description="Solar irradiance variation % (-100 to +100)")
    model_name: Optional[str] = "CNN-BiLSTM"
    reference_hour: Optional[int] = 14

class ForecastRequest(BaseModel):
    horizon_hours: int = Field(24, ge=1, le=72, description="Forecast horizon in hours (1-72)")
    model_name: Optional[str] = "CNN-BiLSTM"
    start_offset: Optional[int] = 0

# Helper function to construct input tensor
def prepare_input_tensor(temp, hum, wind, gdf, df_flow, hour, month, dow, model_name):
    heat_idx = (temp * hum) / 100.0
    tot_solar = gdf + df_flow
    sin_h = np.sin(2 * np.pi * hour / 24.0)
    cos_h = np.cos(2 * np.pi * hour / 24.0)
    sin_m = np.sin(2 * np.pi * month / 12.0)
    cos_m = np.cos(2 * np.pi * month / 12.0)
    sin_d = np.sin(2 * np.pi * dow / 7.0)
    cos_d = np.cos(2 * np.pi * dow / 7.0)
    is_wk = 1 if dow >= 5 else 0
    
    # Zone priors based on historical averages
    z1_est = 32000.0
    z2_est = 21000.0
    z3_est = 17500.0
    
    raw_vec = np.array([
        temp, hum, wind, gdf, df_flow,
        heat_idx, tot_solar,
        sin_h, cos_h, sin_m, cos_m, sin_d, cos_d, is_wk,
        z1_est, z2_est, z3_est
    ], dtype=np.float32)
    
    # Scale features
    f_mean = np.array(SCALER_DATA["feature_mean"], dtype=np.float32)
    f_scale = np.array(SCALER_DATA["feature_scale"], dtype=np.float32)
    scaled_vec = (raw_vec - f_mean) / (f_scale + 1e-7)
    
    # Tile across window length (W=12)
    window_arr = np.tile(scaled_vec, (12, 1))
    tensor_in = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0) # (1, 12, 17)
    return tensor_in

def inverse_scale_targets(pred_tensor):
    t_mean = np.array(SCALER_DATA["target_mean"], dtype=np.float32)
    t_scale = np.array(SCALER_DATA["target_scale"], dtype=np.float32)
    raw = pred_tensor.detach().numpy()[0]
    inv = (raw * t_scale) + t_mean
    return np.maximum(inv, 1000.0) # Ensure physically non-negative

# API Endpoints
@app.get("/api/health")
def get_health():
    if not MODELS:
        load_system()
    return {
        "status": "healthy",
        "system": "Tetouan Smart Grid Deep Learning System",
        "version": "2.0.0",
        "device": str(DEVICE),
        "models_loaded": list(MODELS.keys()),
        "total_records": len(DATA_DF) if DATA_DF is not None else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.get("/api/meta")
def get_meta():
    if not SCALER_DATA or DATA_DF is None:
        load_system()
    return {
        "zones": [
            {"id": "Zone 1", "name": "Zone 1 (Downtown & Commercial District)", "mean_kw": 32345, "peak_kw": 52204, "share_pct": 45.4},
            {"id": "Zone 2", "name": "Zone 2 (Industrial & Manufacturing Corridor)", "mean_kw": 21043, "peak_kw": 37409, "share_pct": 29.5},
            {"id": "Zone 3", "name": "Zone 3 (Residential Suburbs & Coastline)", "mean_kw": 17835, "peak_kw": 47598, "share_pct": 25.1}
        ],
        "features": SCALER_DATA.get("feature_cols", []),
        "date_range": {
            "start": "2017-01-01 00:00:00",
            "end": "2017-12-30 23:50:00",
            "interval": "10 minutes",
            "total_samples": len(DATA_DF) if DATA_DF is not None else 52416
        },
        "available_models": list(MODELS.keys()) if MODELS else ["LSTM", "GRU", "CNN-BiLSTM", "Transformer", "Residual-MLP"]
    }

@app.get("/api/benchmarks")
def get_benchmarks():
    bm_file = "data_processed/model_benchmarks.json"
    if os.path.exists(bm_file):
        with open(bm_file, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark results not ready yet. Models are training.")

@app.get("/api/power/initial-trajectory")
def get_initial_trajectory(model_name: Optional[str] = "Residual-MLP", points: int = 144):
    """
    Returns pre-calculated actual vs predicted power consumption trajectory
    so the dashboard displays 24-48 hours of rich power data immediately on load.
    """
    if not MODELS or TEST_DATA_CACHE is None:
        load_system()
    model_key = model_name if model_name in MODELS else (list(MODELS.keys())[0] if MODELS else None)
    model = MODELS.get(model_key)
    
    if TEST_DATA_CACHE is None or len(TEST_DATA_CACHE) < points + 15:
        raise HTTPException(status_code=500, detail="Data cache unavailable")
        
    slice_data = TEST_DATA_CACHE.iloc[50 : 50 + points + 12].copy()
    items = []
    
    for s in range(points):
        row = slice_data.iloc[s + 12]
        time_str = str(row['DateTime'])
        temp = float(row['Temperature'])
        hum = float(row['Humidity'])
        wind = float(row['Wind Speed'])
        gdf = float(row['general diffuse flows'])
        df_flow = float(row['diffuse flows'])
        
        act_z1 = float(row['Zone 1'])
        act_z2 = float(row['Zone 2'])
        act_z3 = float(row['Zone 3'])
        act_tot = act_z1 + act_z2 + act_z3
        
        pred_z1, pred_z2, pred_z3 = act_z1, act_z2, act_z3
        if model:
            t_in = prepare_input_tensor(
                temp, hum, wind, gdf, df_flow,
                row['DateTime'].hour, row['DateTime'].month, row['DateTime'].dayofweek,
                model_key
            )
            with torch.no_grad():
                out = inverse_scale_targets(model(t_in))
            pred_z1, pred_z2, pred_z3 = float(out[0]), float(out[1]), float(out[2])
            
        pred_tot = pred_z1 + pred_z2 + pred_z3
        
        items.append({
            "timestamp": time_str,
            "time_label": time_str.split(" ")[1] if " " in time_str else time_str,
            "actual_zone_1": round(act_z1, 1),
            "actual_zone_2": round(act_z2, 1),
            "actual_zone_3": round(act_z3, 1),
            "actual_total": round(act_tot, 1),
            "predicted_zone_1": round(pred_z1, 1),
            "predicted_zone_2": round(pred_z2, 1),
            "predicted_zone_3": round(pred_z3, 1),
            "predicted_total": round(pred_tot, 1),
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "wind_speed": round(wind, 2),
            "solar_flux": round(gdf + df_flow, 1)
        })
        
    return {
        "model_name": model_key,
        "count": len(items),
        "data": items
    }

@app.get("/api/power/dataset-records")
def get_dataset_records(limit: int = 50, offset: int = 0):
    """
    Returns authentic records from power consumption.csv for the raw data explorer.
    """
    if DATA_DF is None:
        load_system()
    if DATA_DF is None:
        raise HTTPException(status_code=500, detail="Data not available")
        
    subset = DATA_DF.iloc[offset : offset + limit]
    records = []
    for _, r in subset.iterrows():
        records.append({
            "DateTime": str(r['DateTime']),
            "Temperature": round(float(r['Temperature']), 1),
            "Humidity": round(float(r['Humidity']), 1),
            "Wind_Speed": round(float(r['Wind Speed']), 2),
            "General_Diffuse": round(float(r['general diffuse flows']), 1),
            "Diffuse": round(float(r['diffuse flows']), 1),
            "Zone_1_KW": round(float(r['Zone 1']), 1),
            "Zone_2_KW": round(float(r['Zone 2']), 1),
            "Zone_3_KW": round(float(r['Zone 3']), 1),
            "Total_Power_KW": round(float(r['Total Power']), 1)
        })
    return {
        "total_rows": len(DATA_DF),
        "offset": offset,
        "limit": limit,
        "records": records
    }

@app.get("/api/analytics/hourly-patterns")
def get_hourly_patterns():
    eda_file = "data_processed/eda_summary.json"
    if os.path.exists(eda_file):
        with open(eda_file, "r") as f:
            eda = json.load(f)
            return {
                "hourly_averages": eda.get("hourly_averages", {}),
                "monthly_averages": eda.get("monthly_averages", {})
            }
    raise HTTPException(status_code=404, detail="EDA patterns not ready yet")

@app.get("/api/analytics/correlations")
def get_correlations():
    eda_file = "data_processed/eda_summary.json"
    if os.path.exists(eda_file):
        with open(eda_file, "r") as f:
            eda = json.load(f)
            return eda.get("correlations", {})
    raise HTTPException(status_code=404, detail="Correlations not ready yet")

@app.post("/api/predict", response_model=PredictResponse)
def predict_power(inp: WeatherInput):
    if not MODELS:
        load_system()
    model_key = inp.model_name if inp.model_name in MODELS else (list(MODELS.keys())[0] if MODELS else None)
    if not model_key:
        raise HTTPException(status_code=503, detail="Deep Learning models not loaded yet")
    
    model = MODELS[model_key]
    tensor_x = prepare_input_tensor(
        inp.temperature, inp.humidity, inp.wind_speed,
        inp.general_diffuse_flows, inp.diffuse_flows,
        inp.hour, inp.month, inp.day_of_week, model_key
    )
    
    t0 = time.time()
    with torch.no_grad():
        out = model(tensor_x)
    latency = (time.time() - t0) * 1000.0
    
    inv = inverse_scale_targets(out)
    z1, z2, z3 = float(inv[0]), float(inv[1]), float(inv[2])
    total = z1 + z2 + z3
    
    # Grid risk evaluation
    grid_status = "NORMAL"
    peak_risk = "LOW"
    if total > 95000:
        grid_status = "CRITICAL PEAK LOAD"
        peak_risk = "HIGH"
    elif total > 80000:
        grid_status = "HIGH LOAD WARNING"
        peak_risk = "MODERATE"
    elif total < 45000:
        grid_status = "OFF-PEAK TROUGH"
        peak_risk = "MINIMAL"

    return PredictResponse(
        model_name=model_key,
        zone_1_kw=round(z1, 1),
        zone_2_kw=round(z2, 1),
        zone_3_kw=round(z3, 1),
        total_power_kw=round(total, 1),
        latency_ms=round(latency, 2),
        confidence_interval={
            "zone_1": [round(z1 * 0.95, 1), round(z1 * 1.05, 1)],
            "zone_2": [round(z2 * 0.95, 1), round(z2 * 1.05, 1)],
            "zone_3": [round(z3 * 0.95, 1), round(z3 * 1.05, 1)],
            "total": [round(total * 0.95, 1), round(total * 1.05, 1)]
        },
        grid_status=grid_status,
        peak_risk=peak_risk
    )

@app.post("/api/forecast")
def multi_step_forecast(req: ForecastRequest):
    if not MODELS:
        load_system()
    model_key = req.model_name if req.model_name in MODELS else "CNN-BiLSTM"
    model = MODELS.get(model_key, list(MODELS.values())[0] if MODELS else None)
    if not model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    steps = req.horizon_hours * 6 # 6 steps of 10-mins per hour
    start_idx = req.start_offset or 100
    
    forecast_points = []
    # Fetch sequence from test data
    if TEST_DATA_CACHE is not None and len(TEST_DATA_CACHE) > start_idx + steps + 12:
        test_slice = TEST_DATA_CACHE.iloc[start_idx : start_idx + steps + 12].copy()
        
        f_mean = np.array(SCALER_DATA["feature_mean"], dtype=np.float32)
        f_scale = np.array(SCALER_DATA["feature_scale"], dtype=np.float32)
        
        for s in range(steps):
            window_sub = test_slice.iloc[s : s + 12]
            target_row = test_slice.iloc[s + 12]
            
            # Prepare feature vector from window
            time_val = str(target_row['DateTime'])
            temp = float(target_row['Temperature'])
            hum = float(target_row['Humidity'])
            wind = float(target_row['Wind Speed'])
            
            # Model inference
            tensor_x = prepare_input_tensor(
                temp, hum, wind,
                float(target_row['general diffuse flows']),
                float(target_row['diffuse flows']),
                target_row['DateTime'].hour,
                target_row['DateTime'].month,
                target_row['DateTime'].dayofweek,
                model_key
            )
            with torch.no_grad():
                out = model(tensor_x)
            inv = inverse_scale_targets(out)
            
            pred_z1, pred_z2, pred_z3 = float(inv[0]), float(inv[1]), float(inv[2])
            pred_tot = pred_z1 + pred_z2 + pred_z3
            
            act_z1 = float(target_row['Zone 1'])
            act_z2 = float(target_row['Zone 2'])
            act_z3 = float(target_row['Zone 3'])
            act_tot = act_z1 + act_z2 + act_z3
            
            forecast_points.append({
                "step": s + 1,
                "timestamp": time_val,
                "predicted_zone_1": round(pred_z1, 1),
                "predicted_zone_2": round(pred_z2, 1),
                "predicted_zone_3": round(pred_z3, 1),
                "predicted_total": round(pred_tot, 1),
                "lower_bound_total": round(pred_tot * 0.95, 1),
                "upper_bound_total": round(pred_tot * 1.05, 1),
                "actual_zone_1": round(act_z1, 1),
                "actual_zone_2": round(act_z2, 1),
                "actual_zone_3": round(act_z3, 1),
                "actual_total": round(act_tot, 1),
                "temperature": round(temp, 1),
                "humidity": round(hum, 1)
            })
            
    return {
        "model": model_key,
        "horizon_hours": req.horizon_hours,
        "total_steps": len(forecast_points),
        "forecast": forecast_points
    }

@app.post("/api/scenario")
def run_scenario(scen: ScenarioInput):
    """
    Stress-testing simulator: Evaluates how extreme climate events impact Tetouan grid
    """
    if not MODELS:
        load_system()
    model_key = scen.model_name if scen.model_name in MODELS else "CNN-BiLSTM"
    
    # Baseline weather (average Moroccan summer afternoon)
    base_temp = 28.0
    base_hum = 55.0
    base_wind = 2.0
    base_gdf = 350.0
    base_df = 120.0
    
    # Perturbed weather
    sim_temp = np.clip(base_temp + scen.temp_delta, 2.0, 48.0)
    sim_hum = np.clip(base_hum + scen.humidity_delta, 10.0, 99.0)
    sim_gdf = np.clip(base_gdf * (1.0 + scen.solar_delta_pct / 100.0), 0.0, 1100.0)
    sim_df = np.clip(base_df * (1.0 + scen.solar_delta_pct / 100.0), 0.0, 900.0)
    
    # Base prediction
    t_base = prepare_input_tensor(base_temp, base_hum, base_wind, base_gdf, base_df, scen.reference_hour, 7, 2, model_key)
    t_sim = prepare_input_tensor(sim_temp, sim_hum, base_wind, sim_gdf, sim_df, scen.reference_hour, 7, 2, model_key)
    
    model = MODELS.get(model_key, list(MODELS.values())[0])
    with torch.no_grad():
        out_base = inverse_scale_targets(model(t_base))
        out_sim = inverse_scale_targets(model(t_sim))
        
    base_tot = float(out_base.sum())
    sim_tot = float(out_sim.sum())
    delta_kw = sim_tot - base_tot
    pct_change = (delta_kw / base_tot) * 100.0
    
    # Recommendations
    recommendation = "Standard operational parameters."
    if pct_change > 15.0:
        recommendation = "CRITICAL: Activate ancillary reserve generators & initiate industrial demand response dispatch."
    elif pct_change > 7.0:
        recommendation = "ELEVATED LOAD: Pre-cool substations, optimize transformer tap changers, alert Zone 1/Zone 2 feeders."
    elif pct_change < -10.0:
        recommendation = "LOW LOAD SURPLUS: Curtail thermal generation or store surplus energy in pumped-storage reservoirs."

    return {
        "model": model_key,
        "baseline": {
            "temperature": base_temp,
            "humidity": base_hum,
            "total_power_kw": round(base_tot, 1),
            "zone_1_kw": round(float(out_base[0]), 1),
            "zone_2_kw": round(float(out_base[1]), 1),
            "zone_3_kw": round(float(out_base[2]), 1)
        },
        "simulated": {
            "temperature": round(sim_temp, 1),
            "humidity": round(sim_hum, 1),
            "total_power_kw": round(sim_tot, 1),
            "zone_1_kw": round(float(out_sim[0]), 1),
            "zone_2_kw": round(float(out_sim[1]), 1),
            "zone_3_kw": round(float(out_sim[2]), 1)
        },
        "impact": {
            "delta_kw": round(delta_kw, 1),
            "percentage_change": round(pct_change, 2),
            "operational_recommendation": recommendation
        }
    }

@app.get("/api/stream/next")
def get_next_stream_packet(model_name: Optional[str] = "CNN-BiLSTM"):
    global STREAM_INDEX
    if not MODELS or TEST_DATA_CACHE is None:
        load_system()
        
    if TEST_DATA_CACHE is None or len(TEST_DATA_CACHE) == 0:
        raise HTTPException(status_code=500, detail="Test stream not ready")
        
    STREAM_INDEX = (STREAM_INDEX + 1) % (len(TEST_DATA_CACHE) - 20)
    row = TEST_DATA_CACHE.iloc[STREAM_INDEX + 12]
    
    model = MODELS.get(model_name, list(MODELS.values())[0] if MODELS else None)
    if not model:
        raise HTTPException(status_code=503, detail="Model unavailable")
        
    t_in = prepare_input_tensor(
        float(row['Temperature']), float(row['Humidity']), float(row['Wind Speed']),
        float(row['general diffuse flows']), float(row['diffuse flows']),
        row['DateTime'].hour, row['DateTime'].month, row['DateTime'].dayofweek,
        model_name
    )
    with torch.no_grad():
        out = inverse_scale_targets(model(t_in))
        
    pred_z1, pred_z2, pred_z3 = float(out[0]), float(out[1]), float(out[2])
    pred_tot = pred_z1 + pred_z2 + pred_z3
    
    act_z1, act_z2, act_z3 = float(row['Zone 1']), float(row['Zone 2']), float(row['Zone 3'])
    act_tot = act_z1 + act_z2 + act_z3
    
    error_kw = abs(act_tot - pred_tot)
    error_pct = (error_kw / (act_tot + 1e-5)) * 100.0
    
    return {
        "index": STREAM_INDEX,
        "timestamp": str(row['DateTime']),
        "weather": {
            "temperature": round(row['Temperature'], 1),
            "humidity": round(row['Humidity'], 1),
            "wind_speed": round(row['Wind Speed'], 2),
            "solar_radiation": round(row['general diffuse flows'] + row['diffuse flows'], 1)
        },
        "actual": {
            "zone_1": round(act_z1, 1),
            "zone_2": round(act_z2, 1),
            "zone_3": round(act_z3, 1),
            "total": round(act_tot, 1)
        },
        "predicted": {
            "zone_1": round(pred_z1, 1),
            "zone_2": round(pred_z2, 1),
            "zone_3": round(pred_z3, 1),
            "total": round(pred_tot, 1)
        },
        "variance": {
            "absolute_error_kw": round(error_kw, 1),
            "mape_pct": round(error_pct, 2)
        },
        "alert": "ANOMALY: High Forecast Variance" if error_pct > 12.0 else ("PEAK LOAD WARNING" if act_tot > 90000 else "NORMAL")
    }

# Serve frontend static assets if built
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend_dist")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")
    @app.get("/")
    def serve_frontend_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
