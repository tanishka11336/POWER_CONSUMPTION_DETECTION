# Tetouan Smart Grid Deep Learning Multi-Zone Power Consumption Forecasting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Model Accuracy](https://img.shields.io/badge/Best%20R%C2%B2-0.9964%20(99.64%25)-success.svg)]()
[![Best MAPE](https://img.shields.io/badge/Best%20MAPE-1.06%25-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-0.004%20ms-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade, end-to-end Deep Learning & Operational Intelligence platform for multi-zone electrical power demand forecasting, load anomaly detection, and climate-stress simulation across the distribution network of **Tetouan City, Morocco**.

---

## Table of Contents
1. [Executive Abstract & Research Summary](#1-executive-abstract--research-summary)
2. [Project Motivation & Grid Topology](#2-project-motivation--grid-topology)
3. [Dataset Architecture & Exploratory Data Analysis (EDA)](#3-dataset-architecture--exploratory-data-analysis-eda)
4. [Feature Engineering & Preprocessing Pipeline](#4-feature-engineering--preprocessing-pipeline)
5. [Deep Learning Model Architectures](#5-deep-learning-model-architectures)
6. [Experimental Results & Empirical Benchmarks](#6-experimental-results--empirical-benchmarks)
7. [System Architecture & Full-Stack Deployment](#7-system-architecture--full-stack-deployment)
8. [REST API Documentation & Endpoints](#8-rest-api-documentation--endpoints)
9. [Step-by-Step Execution Guide](#9-step-by-step-execution-guide)
10. [Academic & Professional Report Writing Guide](#10-academic--professional-report-writing-guide)
11. [Project Directory Structure](#11-project-directory-structure)

---

## 1. Executive Abstract & Research Summary

> **Ready-to-Use Abstract for Academic Papers, Project Reports, and Theses:**
>
> Modern electrical transmission and distribution grids face unprecedented dispatch volatility driven by rapid urban expansion, climate variability, and decentralized load dynamics. Accurate, multi-horizon load forecasting is vital to prevent transmission line congestion, avert transformer overloading, optimize spinning reserves, and lower peak generation costs. This study presents a production-grade multi-zone power consumption forecasting framework benchmarked on **52,416 high-resolution micro-interval telemetry records** (10-minute cadence over a full calendar year) from the municipal grid of **Tetouan, Morocco**.
>
> Five distinct deep learning paradigms were designed, implemented, and empirically compared under a strict leak-free chronological holdout evaluation protocol: (1) **Residual Deep MLP (Dense ResNet)**, (2) **Long Short-Term Memory (LSTM)**, (3) **Gated Recurrent Unit (GRU)**, (4) **Temporal Transformer (Self-Attention)**, and (5) **1D-CNN + Bidirectional LSTM (Hybrid Spatio-Temporal)**. Features incorporate cyclical calendar harmonics ($\sin/\cos$), thermodynamic heat-index approximations, and cumulative solar irradiance.
>
> Experimental results establish that the **Residual Deep MLP** delivers state-of-the-art predictive performance, achieving an **$R^2$ score of 0.9964 (99.64% variance explained)**, a **Mean Absolute Error (MAE) of 640.93 KW**, a **Root Mean Squared Error (RMSE) of 853.09 KW**, and a **Mean Absolute Percentage Error (MAPE) of 1.06%** across aggregate grid load. The model executes inference in **0.004 milliseconds per sequence**, enabling sub-millisecond edge SCADA/EMS dispatch loops. The complete system is deployed as a high-throughput **FastAPI** service coupled with an interactive web dashboard providing real-time telemetry streaming, recursive multi-horizon projections (1h–48h), and climate-sensitivity stress sandboxes.

### Key Highlights & Contributions
- **Full-Spectrum Neural Comparison**: Benchmark of 5 deep learning architectures on the identical chronological test set.
- **Microsecond Serving Latency**: Sub-millisecond CPU inference across all architectures, with Residual-MLP leading at $4\ \mu\text{s}$ per sample.
- **Physical Feature Synergy**: Integration of thermodynamic heat index proxies and solar radiation vectors with diurnal periodicity.
- **Full-Stack Production System**: Turnkey FastAPI backend and Cyber-Grid telemetry dashboard with zero external cloud dependencies.

---

## 2. Project Motivation & Grid Topology

### The Smart Grid Challenge
Grid dispatchers must balance supply and demand in real time. Overestimating demand leads to costly idling of reserve peaking plants, while underestimating demand risks brownouts, transformer tripping, and voltage degradation.

### Tetouan City Distribution Zones
Tetouan's distribution network exhibits three distinct socio-economic and topological load profiles:

```
+------------------------------------------------------------------------------------------+
|                               TETOUAN MUNICIPAL SMART GRID                               |
+-----------------------------------+----------------------------------+-------------------+
|              ZONE 1               |              ZONE 2              |      ZONE 3       |
|    Downtown Commercial & Admin    | Industrial Parks & Manufacturing | Suburbs & Tourism |
|  - High baseline day demand       | - Steady weekday factory shifts  | - Highly variable |
|  - Predictable business hours     | - Moderate weekend drops         | - Tourism spikes  |
|  - Share: 45.4% of total grid     | - Share: 29.5% of total grid     | - Share: 25.1%    |
+-----------------------------------+----------------------------------+-------------------+
```

1. **Zone 1 (Downtown Commercial & Administrative):**
   - Dominant consumer ($32,345\text{ KW}$ mean load; $45.4\%$ total demand).
   - Driven by retail, municipal offices, lighting, and commercial air conditioning.
2. **Zone 2 (Industrial Parks & Logistics Corridor):**
   - Secondary consumer ($21,043\text{ KW}$ mean load; $29.5\%$ total demand).
   - Characterized by shift-based manufacturing machinery and industrial motors.
3. **Zone 3 (Residential Suburbs & Coastal Tourism Belt):**
   - High-variance consumer ($17,835\text{ KW}$ mean load; $25.1\%$ total demand).
   - Driven by domestic household appliances, decentralized HVAC, and seasonal tourist influx.

---

## 3. Dataset Architecture & Exploratory Data Analysis (EDA)

The telemetry source (`power consumption.csv`) comprises continuous sensor data from **January 1, 2017 00:00:00** to **December 30, 2017 23:50:00** sampled every 10 minutes ($6 \times 24 \times 364 = 52,416$ time steps).

### 3.1 Zone Power Consumption Distribution
| Zone Identifier | Functional Type | Mean (KW) | Std Dev (KW) | Min (KW) | Peak Load (KW) | Grid Share (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Zone 1** | Downtown Commercial & Admin | 32,344.97 | 7,130.56 | 13,895.70 | 52,204.40 | 45.4% |
| **Zone 2** | Industrial Parks & Manufacturing | 21,042.51 | 5,201.47 | 8,560.08 | 37,408.86 | 29.5% |
| **Zone 3** | Residential Suburbs & Tourism | 17,835.41 | 6,622.17 | 5,935.17 | 47,598.32 | 25.1% |
| **Total Grid** | **Aggregate Municipal Demand** | **71,222.89** | **17,143.14** | **36,785.04** | **134,208.15** | **100.0%** |

### 3.2 Meteorological Covariates
| Sensor Variable | Unit | Mean | Min | Max | Std Dev | Correlation with Total Power ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temperature** | $^\circ\text{C}$ | 18.81 | 3.25 | 40.01 | 5.82 | **+0.4882** (Strong Positive) |
| **Relative Humidity** | % | 68.26 | 11.34 | 94.80 | 15.55 | **-0.2991** (Moderate Negative) |
| **Wind Speed** | m/s | 1.96 | 0.05 | 6.48 | 2.35 | **+0.2217** (Positive) |
| **General Diffuse Flows** | $\text{W/m}^2$ | 182.70 | 0.00 | 1,163.00 | 264.40 | **+0.1880** (Positive) |
| **Diffuse Flows** | $\text{W/m}^2$ | 75.03 | 0.00 | 936.00 | 123.90 | **+0.1504** (Positive) |

### 3.3 Correlation Matrix
$$\begin{pmatrix}
 & \text{Temp} & \text{Humidity} & \text{Wind} & \text{Solar} & \text{Zone 1} & \text{Zone 2} & \text{Zone 3} & \mathbf{Total} \\
\text{Temp} & 1.000 & -0.460 & +0.477 & +0.460 & +0.440 & +0.382 & +0.490 & \mathbf{+0.488} \\
\text{Humidity} & -0.460 & 1.000 & -0.136 & -0.468 & -0.287 & -0.295 & -0.233 & \mathbf{-0.299} \\
\text{Wind} & +0.477 & -0.136 & 1.000 & +0.134 & +0.167 & +0.146 & +0.279 & \mathbf{+0.222} \\
\text{Solar} & +0.460 & -0.468 & +0.134 & 1.000 & +0.188 & +0.157 & +0.063 & \mathbf{+0.150} \\
\mathbf{Total} & \mathbf{+0.488} & \mathbf{-0.299} & \mathbf{+0.222} & \mathbf{+0.150} & \mathbf{+0.876} & \mathbf{+0.832} & \mathbf{+0.814} & \mathbf{1.000}
\end{pmatrix}$$

---

## 4. Feature Engineering & Preprocessing Pipeline

### 4.1 Chronological Partitioning (Zero Data Leakage)
To prevent temporal data leakage, standard random k-fold shuffling is strictly avoided. The continuous dataset is split sequentially:
- **Training Partition (70%):** Index $0 \to 36,691$ (Jan 1, 2017 to Sep 13, 2017)
- **Validation Partition (15%):** Index $36,692 \to 44,553$ (Sep 13, 2017 to Nov 6, 2017)
- **Test Holdout (15%):** Index $44,554 \to 52,416$ (Nov 6, 2017 to Dec 30, 2017 — 7,851 sequences)

### 4.2 17-Dimensional Engineered Feature Vector
Each time step $t$ is projected into a 17-dimensional feature space $\mathbf{x}_t \in \mathbb{R}^{17}$:

1. **Exogenous Meteorological Telemetry (5 dims):**
   - Ambient Temperature ($T$), Relative Humidity ($H$), Wind Speed ($WS$), General Diffuse Flows ($GDF$), Diffuse Flows ($DF$).
2. **Thermodynamic Physical Proxies (2 dims):**
   $$\text{HeatIndexProxy} = \frac{T \times H}{100.0}$$
   $$\text{TotalSolarFlux} = GDF + DF$$
3. **Cyclical Periodic Encodings (6 dims):**
   $$\sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
   $$\sin\left(\frac{2\pi \cdot \text{Month}}{12}\right), \quad \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)$$
   $$\sin\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$$
4. **Calendar Indicator (1 dim):**
   $$\text{IsWeekend} \in \{0, 1\}$$
5. **Autoregressive State Vectors (3 dims):**
   - Historical consumption loads for Zone 1, Zone 2, and Zone 3.

### 4.3 Temporal Sliding Window Construction
- **Lookback Window ($W$):** 12 consecutive 10-minute intervals ($W = 12 \implies 2\text{ hours}$ lookback).
- **Tensor Input:** $\mathbf{X} \in \mathbb{R}^{B \times 12 \times 17}$ where $B$ is batch size.
- **Target Output:** $\mathbf{Y} \in \mathbb{R}^{B \times 3}$ predicting $[\text{Zone 1}, \text{Zone 2}, \text{Zone 3}]$ at time $t+1$.
- **Standardization:** Input features and target variables are normalized using `StandardScaler` fitted strictly on the training partition:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 5. Deep Learning Model Architectures

All models are defined in [backend/models_definition.py](file:///c:/Users/Tanishka/Desktop/MID_TERM_ADSL/backend/models_definition.py) and trained with PyTorch using Mean Squared Error (MSE) loss and the Adam optimizer.

### 5.1 Residual Deep MLP (ResNet-MLP) — Top Performer
- **Architecture Philosophy:** Unrolls the $12 \times 17$ sequence into a dense vector ($204$ features), mapping it through deep linear layers with skip connections, Layer Normalization, and GELU non-linearities.
- **Formulation:**
  $$\mathbf{h}_0 = \mathbf{W}_{\text{in}} \cdot \text{vec}(\mathbf{X}) + \mathbf{b}_{\text{in}}$$
  $$\mathbf{h}_1 = \mathbf{h}_0 + \text{Dropout}\Big(\text{GELU}\big(\mathbf{W}_1 \cdot \text{LayerNorm}(\mathbf{h}_0) + \mathbf{b}_1\big)\Big)$$
  $$\mathbf{h}_2 = \mathbf{h}_1 + \text{Dropout}\Big(\text{GELU}\big(\mathbf{W}_2 \cdot \text{LayerNorm}(\mathbf{h}_1) + \mathbf{b}_2\big)\Big)$$
  $$\hat{\mathbf{Y}} = \mathbf{W}_{\text{out}} \cdot \mathbf{h}_2 + \mathbf{b}_{\text{out}}$$
- **Parameters:** 60,163 | **Latency:** 0.004 ms/sample

### 5.2 Long Short-Term Memory (LSTM)
- **Architecture Philosophy:** 2-layer recurrent network maintaining cell state $\mathbf{c}_t$ and hidden state $\mathbf{h}_t$ via forget, input, and output gates to mitigate vanishing gradients over sequential windows.
- **Formulation:**
  $$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f), \quad \mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$
  $$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c), \quad \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
  $$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o), \quad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$
- **Parameters:** 56,707 | **Latency:** 0.065 ms/sample

### 5.3 Gated Recurrent Unit (GRU)
- **Architecture Philosophy:** Streamlined recurrent structure coupling input and forget mechanics into reset ($\mathbf{r}_t$) and update ($\mathbf{z}_t$) gates, eliminating the separate memory cell for faster convergence.
- **Formulation:**
  $$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z), \quad \mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r)$$
  $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h)$$
  $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
- **Parameters:** 43,075 | **Latency:** 0.065 ms/sample

### 5.4 Temporal Transformer (Self-Attention Network)
- **Architecture Philosophy:** 4-head scaled dot-product self-attention mechanism with learnable positional embeddings, processing entire sequence intervals in parallel without recurrence bottlenecks.
- **Formulation:**
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
- **Parameters:** 70,275 | **Latency:** 0.025 ms/sample

### 5.5 1D-CNN + Bidirectional LSTM (Hybrid)
- **Architecture Philosophy:** 1D convolution layer ($k=3$, 64 filters) with Batch Normalization extracts local morphological gradient transitions, passed into a Bidirectional LSTM to track forward and backward context.
- **Parameters:** 30,723 | **Latency:** 0.046 ms/sample

---

## 6. Experimental Results & Empirical Benchmarks

### 6.1 Comprehensive Aggregate Grid Benchmark
Evaluated on **7,851 unseen chronological test sequences** (Nov 6, 2017 to Dec 30, 2017):

| Model Architecture | Trainable Params | Total Power $R^2$ | Total Power RMSE (KW) | Total Power MAE (KW) | Total Power MAPE (%) | Latency (ms/sample) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Residual-MLP** 🏆 | **60,163** | **0.9964** | **853.09** | **640.93** | **1.06%** | **0.004 ms** | **14.5s** |
| **LSTM** | 56,707 | 0.9873 | 1,605.74 | 1,301.92 | 2.05% | 0.065 ms | 47.8s |
| **GRU** | 43,075 | 0.9836 | 1,823.03 | 1,580.88 | 2.68% | 0.065 ms | 74.0s |
| **Transformer** | 70,275 | 0.9824 | 1,889.89 | 1,529.60 | 2.48% | 0.025 ms | 97.8s |
| **CNN-BiLSTM** | 30,723 | 0.9751 | 2,247.56 | 1,960.72 | 3.34% | 0.046 ms | 38.9s |

### 6.2 Zone-Specific Breakdown for Residual-MLP (Best Model)
| Grid Zone | Test $R^2$ Score | RMSE (KW) | MAE (KW) | MAPE (%) | Characterization |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Zone 1 (Downtown)** | **0.9945** | **448.30** | **334.97** | **1.25%** | Highly structured commercial routine; minimal error |
| **Zone 2 (Industrial)** | **0.9919** | **504.60** | **367.32** | **1.50%** | Predictable manufacturing shifts and steady load |
| **Zone 3 (Residential)**| **0.9715** | **567.25** | **473.85** | **4.22%** | High stochastic consumer variance & tourism spikes |
| **Total Grid Aggregate**| **0.9964** | **853.09** | **640.93** | **1.06%** | Exceptional multi-zone balancing stability |

### 6.3 Deep Engineering & Scientific Insights
1. **Why Residual-MLP Outperforms Recurrent Networks:**
   At 10-minute sampling intervals, the system displays strong short-range temporal autocorrelation and diurnal smoothness. Recurrent networks (LSTM/GRU) process sequential tokens recursively, propagating small gate inaccuracies. In contrast, the unrolled Residual-MLP applies direct affine projections with residual highway skips, mitigating gradient attenuation and capturing inter-timestep dependencies in $O(1)$ operations.
2. **Zone Error Variance Analysis:**
   Zone 1 achieves near-perfect predictability ($R^2 = 0.9945$, $\text{MAPE} = 1.25\%$) because commercial districts follow strictly scheduled operating regimes. Conversely, Zone 3 has higher percentage deviation ($\text{MAPE} = 4.22\%$) because residential energy consumption incorporates individual human behaviors and unregulated HVAC cycling.
3. **Serving Latency & Edge SCADA Feasibility:**
   The Residual-MLP's inference latency of $4\ \mu\text{s}$ ($0.004\text{ ms}$) on standard CPU hardware demonstrates that complex neural models can easily be deployed on low-cost substation edge devices (e.g., Raspberry Pi, RTUs, PLCs) without dedicated GPUs.

---

## 7. System Architecture & Full-Stack Deployment

The solution is architected as an integrated cyber-grid telemetry and analytics microservice:

```
+-----------------------------------------------------------------------------------+
|                        CLIENT DASHBOARD LAYER (HTML5 / JS / CSS)                  |
|  - Real-Time Live Telemetry Stream Canvas & Channel Toggles (Total, Z1, Z2, Z3)   |
|  - Multi-Horizon Forecasting Engine (1h, 6h, 12h, 24h, 48h Recursive Lookahead)   |
|  - "What-If" Climate & Load Stress Sandbox (Sliders: Temp, Humidity, Solar Flux)  |
|  - Deep Learning Benchmark Hub (Interactive Radar & Bar Visualizations)          |
|  - Dataset Browser & Zone Topology Visualizer                                     |
+-----------------------------------------------------------------------------------+
                                       |
                   REST HTTP / JSON Invocations (Port 8000)
                                       v
+-----------------------------------------------------------------------------------+
|                           FASTAPI SERVING BACKEND                                 |
|  - /api/health                   : Health status, model registry, device info     |
|  - /api/benchmarks               : Empirical test metrics & loss history          |
|  - /api/predict                  : Single/sequence inference + confidence bounds  |
|  - /api/forecast                 : Multi-step autoregressive recursive rollout    |
|  - /api/scenario                 : Climate sensitivity stress simulation          |
|  - /api/stream/next              : High-frequency telemetry stream generator      |
|  - /api/analytics/*              : Downsampled historical & diurnal profiles      |
+-----------------------------------------------------------------------------------+
                                       |
                     In-Memory Tensor Preprocessing & Scaling
                                       v
+-----------------------------------------------------------------------------------+
|                     PYTORCH NEURAL INFERENCE ENGINE (CPU)                         |
|  - Residual-MLP Checkpoint       : models/residual_mlp_best.pt  (1.06% MAPE)      |
|  - LSTM Checkpoint               : models/lstm_best.pt          (2.05% MAPE)      |
|  - GRU Checkpoint                : models/gru_best.pt           (2.68% MAPE)      |
|  - Transformer Checkpoint        : models/transformer_best.pt   (2.48% MAPE)      |
|  - CNN-BiLSTM Checkpoint         : models/cnn_bilstm_best.pt    (3.34% MAPE)      |
+-----------------------------------------------------------------------------------+
```

---

## 8. REST API Documentation & Endpoints

When the server runs, interactive OpenAPI / Swagger documentation is available at:
`http://localhost:8000/docs`

### Major API Endpoints
| Endpoint | Method | Description | Sample Request / Query |
| :--- | :---: | :--- | :--- |
| `/api/health` | `GET` | Health check, loaded models, dataset status | None |
| `/api/benchmarks` | `GET` | Full test metrics, loss histories, zone tables | None |
| `/api/predict` | `POST` | Real-time single-step inference + confidence intervals | `{"temperature": 26.5, "humidity": 60, "model_name": "Residual-MLP"}` |
| `/api/forecast` | `GET` | Recursive multi-step forward forecast | `?steps=24&model_name=Residual-MLP` |
| `/api/scenario` | `POST` | Climate & load stress "What-If" simulation | `{"temp_delta": 5.0, "humidity_delta": -10.0}` |
| `/api/stream/next` | `GET` | Next sequential telemetry step from test cache | `?model_name=Residual-MLP` |
| `/api/stream/reset`| `POST`| Resets live telemetry stream index to 0 | None |
| `/api/analytics/diurnal` | `GET` | 24-hour diurnal load curves per zone | None |
| `/api/dataset/sample` | `GET` | Paginated records from raw 52,416-row dataset | `?offset=0&limit=50` |

---

## 9. Step-by-Step Execution Guide

### Prerequisites
- Python 3.10 or higher
- PowerShell, CMD, or Bash

### 9.1 Quick Start (Recommended)
All neural checkpoints and data summaries are already pre-computed and stored in `models/` and `data_processed/`. Simply run:

```powershell
# 1. Navigate to the project root directory
cd c:\Users\Tanishka\Desktop\MID_TERM_ADSL

# 2. (First-time only) Install required dependencies
pip install -r requirements.txt

# 3. Launch the unified FastAPI Server & Telemetry Dashboard
python run_server.py
```

Open your browser:
- **Interactive Control Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 9.2 Complete Pipeline Execution from Scratch (End-to-End)
To re-verify EDA, re-train all 5 deep neural networks from the raw dataset, and deploy:

```powershell
# Step 1: Execute Exploratory Data Analysis (EDA)
# Reads 'power consumption.csv', computes descriptive stats, saves 'data_processed/eda_summary.json'
python scripts/eda.py

# Step 2: Train & Benchmark All 5 Deep Learning Models
# Trains Residual-MLP, LSTM, GRU, Transformer, CNN-BiLSTM
# Exports scalers, best .pt model weights, and performance evaluation metrics
python scripts/train_models.py

# Step 3: Launch Production Web Server
python run_server.py
```

---

## 10. Academic & Professional Report Writing Guide

This section provides ready structured materials for mid-term project submissions, final thesis chapters, or technical presentations:

### 10.1 Key Mathematical Formulations for Reports
- **Mean Absolute Percentage Error (MAPE):**
  $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

### 10.2 Recommended Report Outline
1. **Introduction & Motivation:** Power demand challenges in municipal microgrids; objectives of multi-zone load forecasting.
2. **Related Work:** Traditional ARIMA/SARIMAX vs modern Deep Neural Networks (RNN, LSTM, Transformers).
3. **Dataset & Feature Engineering:** Tetouan 52,416-record topology, weather covariates, cyclical encodings, sliding window creation.
4. **Methodology:** Detailed architectures of Residual-MLP, LSTM, GRU, Temporal Transformer, and CNN-BiLSTM.
5. **Experimental Results & Discussion:** Performance comparison table, error analysis across Zones 1, 2, and 3, latency vs accuracy trade-offs.
6. **System Implementation:** FastAPI serving architecture, real-time telemetry streaming, and UI dashboard features.
7. **Conclusion & Future Scope:** Edge deployment, solar PV integration, reinforcement-learning-based dispatch.

### 10.3 Slide Presentation Talking Points (Defense / Mid-Term)
- **Slide 1 (Problem):** Multi-zone municipal load is non-linear and climate-dependent; inaccurate forecasting causes grid imbalance and high costs.
- **Slide 2 (Data):** 1-year continuous dataset (52,416 records, 10-minute cadence) across downtown, industrial, and residential sectors.
- **Slide 3 (Innovation):** 17 engineered features combining cyclical trigonometric harmonics, solar irradiance, and heat index proxies.
- **Slide 4 (Results):** Residual-MLP achieves 99.64% $R^2$ and 1.06% MAPE, outperforming complex recurrent/transformer networks while running in $4\ \mu\text{s}$.
- **Slide 5 (Live System):** Full-stack web dashboard demonstrating live telemetry streaming, recursive multi-horizon lookahead, and climate stress sandboxes.

---

## 11. Project Directory Structure

```
MID_TERM_ADSL/
├── backend/
│   ├── main.py                     # FastAPI backend application & API routing
│   └── models_definition.py        # PyTorch implementations of all 5 neural models
├── data_processed/
│   ├── eda_summary.json            # Statistical summary and correlation profiles
│   ├── model_benchmarks.json       # Empirical benchmark results & training loss history
│   ├── sample_stream.csv           # Telemetry stream cache for real-time simulation
│   └── *_sample_preds.json         # Model-specific test predictions
├── frontend_dist/
│   ├── index.html                  # Cyber-grid responsive single-page web dashboard
│   └── static/
│       ├── css/
│       │   └── style.css           # Premium dark-theme CSS styling system
│       └── js/
│           ├── app.js              # Real-time telemetry streaming & Chart.js logic
│           └── chart.umd.min.js    # Local offline Chart.js visualization engine
├── models/
│   ├── cnn_bilstm_best.pt          # Saved checkpoint: CNN-BiLSTM
│   ├── gru_best.pt                 # Saved checkpoint: GRU
│   ├── lstm_best.pt                # Saved checkpoint: LSTM
│   ├── residual_mlp_best.pt        # Saved checkpoint: Residual-MLP (Best Model)
│   ├── transformer_best.pt         # Saved checkpoint: Transformer
│   └── scalers.json                # Preprocessing feature & target scalers
├── reports/
│   ├── figures/                    # Auto-generated model training & evaluation plots
│   └── PRODUCTION_REPORT.md        # Comprehensive technical report document
├── scripts/
│   ├── eda.py                      # Exploratory data analysis extraction script
│   └── train_models.py             # Model training, benchmarking & figure generation script
├── power consumption.csv          # Raw municipal telemetry dataset (52,416 rows)
├── requirements.txt                # Python project dependencies
├── run_server.py                   # Production launcher script
└── README.md                       # Complete documentation & project manual
```

---

## 12. Authors & Acknowledgments
- **Project:** Tetouan City Smart Grid Deep Learning Power Consumption Forecasting
- **Dataset Source:** Telemetry from the municipal power distribution authority of Tetouan, Morocco (UCI Machine Learning Repository).
- **Architecture & Engineering:** Production Deep Learning & AI Grid Optimization Team.
