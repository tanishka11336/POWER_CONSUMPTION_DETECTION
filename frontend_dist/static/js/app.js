// Tetouan City Power Consumption Deep Learning Control Center - Client Application
const API_BASE = window.location.origin;

// State management
let state = {
  currentModel: "Residual-MLP",
  availableModels: ["Residual-MLP", "LSTM", "GRU", "Transformer", "CNN-BiLSTM"],
  benchmarks: null,
  isStreaming: false,
  streamInterval: null,
  streamSpeedMs: 1500,
  datasetOffset: 0,
  datasetLimit: 50,
  charts: {}
};

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupEventListeners();
  setupCalculatorListeners();
  setupChannelToggles();
  await checkHealthAndMeta();
  await loadBenchmarks();
  await initializeCharts();
  await loadInitialPowerTrajectory();
  await loadHourlyPatterns();
  await loadDatasetRecords(0);
  triggerCalculatorPrediction();
});

// Navigation Handling
function setupNavigation() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
      
      btn.classList.add("active");
      const targetContent = document.getElementById(`tab-${target}`);
      if (targetContent) {
        targetContent.classList.add("active");
      }
      
      setTimeout(() => {
        Object.values(state.charts).forEach(c => c && c.resize && c.resize());
      }, 100);
    });
  });
}

function setupEventListeners() {
  const modelSelect = document.getElementById("model-select");
  if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
      state.currentModel = e.target.value;
      updateModelInfoBadge();
      loadInitialPowerTrajectory();
      triggerForecastUpdate();
      triggerCalculatorPrediction();
    });
  }

  const streamBtn = document.getElementById("toggle-stream-btn");
  if (streamBtn) streamBtn.addEventListener("click", toggleStreaming);

  const streamStepBtn = document.getElementById("step-stream-btn");
  if (streamStepBtn) streamStepBtn.addEventListener("click", fetchStreamStep);

  const horizonSelect = document.getElementById("forecast-horizon");
  if (horizonSelect) horizonSelect.addEventListener("change", triggerForecastUpdate);

  const prevBtn = document.getElementById("dataset-prev-btn");
  const nextBtn = document.getElementById("dataset-next-btn");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.datasetOffset >= state.datasetLimit) {
        state.datasetOffset -= state.datasetLimit;
        loadDatasetRecords(state.datasetOffset);
      }
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      state.datasetOffset += state.datasetLimit;
      loadDatasetRecords(state.datasetOffset);
    });
  }
}

function setupChannelToggles() {
  const chs = [
    { id: "toggle-ch-total", idx: 0 },
    { id: "toggle-ch-pred", idx: 1 },
    { id: "toggle-ch-z1", idx: 2 },
    { id: "toggle-ch-z2", idx: 3 },
    { id: "toggle-ch-z3", idx: 4 }
  ];
  chs.forEach(ch => {
    const el = document.getElementById(ch.id);
    if (el) {
      el.addEventListener("change", (e) => {
        const chart = state.charts.liveStream;
        if (chart && chart.data.datasets[ch.idx]) {
          chart.data.datasets[ch.idx].hidden = !e.target.checked;
          chart.update();
        }
      });
    }
  });
}

// System Health & Metadata
async function checkHealthAndMeta() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return;
    const data = await res.json();
    const statusText = document.getElementById("system-status-text");
    if (statusText) {
      statusText.innerText = `ACTIVE // ${data.device.toUpperCase()}`;
    }
    
    if (data.models_loaded && data.models_loaded.length > 0) {
      state.availableModels = data.models_loaded;
      const modelSelect = document.getElementById("model-select");
      if (modelSelect && modelSelect.options.length <= 1) {
        modelSelect.innerHTML = "";
        data.models_loaded.forEach(m => {
          const opt = document.createElement("option");
          opt.value = m;
          opt.innerText = `${m} Deep Model`;
          if (m === state.currentModel) opt.selected = true;
          modelSelect.appendChild(opt);
        });
      }
    }
  } catch (err) {
    console.warn("API health check:", err);
  }
}

function updateModelInfoBadge() {
  const badge = document.getElementById("active-model-badge");
  if (badge) badge.innerText = state.currentModel;
  
  if (state.benchmarks && state.benchmarks[state.currentModel]) {
    const bm = state.benchmarks[state.currentModel];
    const r2El = document.getElementById("current-model-r2");
    const rmseEl = document.getElementById("current-model-rmse");
    const maeEl = document.getElementById("current-model-mae");
    const latEl = document.getElementById("current-model-latency");
    if (r2El) r2El.innerText = `${bm["Total Power"].R2.toFixed(4)} (${(bm["Total Power"].R2 * 100).toFixed(2)}%)`;
    if (rmseEl) rmseEl.innerText = `${bm["Total Power"].RMSE.toFixed(1)} KW`;
    if (maeEl) maeEl.innerText = `${bm["Total Power"].MAE.toFixed(1)} KW (${bm["Total Power"].MAPE.toFixed(2)}% MAPE)`;
    if (latEl) latEl.innerText = `${bm.latency_ms_per_sample.toFixed(3)} ms`;
  }
}

// Load Benchmark Data & Render Comparison Matrix
async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks`);
    if (!res.ok) return;
    state.benchmarks = await res.json();
    
    renderBenchmarkTable();
    renderBenchmarkCharts();
    updateModelInfoBadge();
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

function renderBenchmarkTable() {
  const tbody = document.getElementById("benchmark-table-body");
  if (!tbody || !state.benchmarks) return;
  tbody.innerHTML = "";

  const sortedModels = Object.keys(state.benchmarks).sort((a, b) => 
    state.benchmarks[b]["Total Power"].R2 - state.benchmarks[a]["Total Power"].R2
  );

  sortedModels.forEach((m, idx) => {
    const data = state.benchmarks[m];
    const tot = data["Total Power"];
    const z1 = data.zones["Zone 1"];
    const z2 = data.zones["Zone 2"];
    const z3 = data.zones["Zone 3"];
    
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>
        <strong>${m}</strong>
        ${idx === 0 ? '<span class="badge-tag badge-green" style="margin-left: 8px;">TOP ACCURACY</span>' : ''}
      </td>
      <td>${data.parameters.toLocaleString()}</td>
      <td class="${idx === 0 ? 'best-badge' : ''}">${tot.R2.toFixed(4)}</td>
      <td>${tot.RMSE.toFixed(1)} KW</td>
      <td>${tot.MAE.toFixed(1)} KW</td>
      <td>${tot.MAPE.toFixed(2)}%</td>
      <td>${z1.RMSE.toFixed(1)} / ${z2.RMSE.toFixed(1)} / ${z3.RMSE.toFixed(1)}</td>
      <td>${data.latency_ms_per_sample.toFixed(3)} ms</td>
      <td>${data.training_time_sec.toFixed(1)}s</td>
    `;
    tbody.appendChild(row);
  });
}

// Initialize Interactive Charts
async function initializeCharts() {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not available yet");
    return;
  }

  // 1. Main Power Consumption Time Series
  const ctxLive = document.getElementById("chart-live-stream");
  if (ctxLive) {
    state.charts.liveStream = new Chart(ctxLive, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Total City Power Consumption (KW)",
            data: [],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.08)",
            borderWidth: 2.8,
            fill: true,
            pointRadius: 0,
            tension: 0.3
          },
          {
            label: "AI Predicted Power (KW)",
            data: [],
            borderColor: "#06b6d4",
            borderDash: [5, 4],
            borderWidth: 2.2,
            pointRadius: 0,
            tension: 0.3,
            fill: false
          },
          {
            label: "Zone 1 Power (Downtown KW)",
            data: [],
            borderColor: "#3b82f6",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.3,
            fill: false
          },
          {
            label: "Zone 2 Power (Industrial KW)",
            data: [],
            borderColor: "#f59e0b",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.3,
            fill: false
          },
          {
            label: "Zone 3 Power (Residential KW)",
            data: [],
            borderColor: "#8b5cf6",
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.3,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "#9ca3af", font: { family: "Inter", size: 12 } } },
          tooltip: {
            backgroundColor: "rgba(17, 24, 39, 0.95)",
            titleColor: "#f9fafb",
            bodyColor: "#38bdf8",
            borderColor: "rgba(255, 255, 255, 0.1)",
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                return `${context.dataset.label}: ${Math.round(context.raw).toLocaleString()} KW`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#6b7280", maxTicksLimit: 14 }
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#9ca3af",
              callback: function(v) { return v >= 1000 ? (v / 1000).toFixed(0) + " MW" : v + " KW"; }
            },
            title: { display: true, text: "Power Demand (KW / MW)", color: "#9ca3af" }
          }
        }
      }
    });
  }

  // 2. Multi Horizon Forecast Chart
  const ctxForecast = document.getElementById("chart-forecast-horizon");
  if (ctxForecast) {
    state.charts.forecast = new Chart(ctxForecast, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Ground Truth Total Power (KW)",
            data: [],
            borderColor: "#64748b",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3
          },
          {
            label: "Forecast Power Projection (KW)",
            data: [],
            borderColor: "#38bdf8",
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3
          },
          {
            label: "95% Upper Bound",
            data: [],
            borderColor: "rgba(56, 189, 248, 0.2)",
            borderWidth: 1,
            pointRadius: 0,
            fill: "+1",
            backgroundColor: "rgba(56, 189, 248, 0.12)"
          },
          {
            label: "95% Lower Bound",
            data: [],
            borderColor: "rgba(56, 189, 248, 0.2)",
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { color: "#9ca3af" } } },
        scales: {
          x: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#6b7280", maxTicksLimit: 14 } },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af", callback: function(v) { return (v / 1000).toFixed(0) + " MW"; } }
          }
        }
      }
    });
  }

  // 3. Zone Stacked Load Chart
  const ctxZoneStacked = document.getElementById("chart-zone-stacked");
  if (ctxZoneStacked) {
    state.charts.zoneStacked = new Chart(ctxZoneStacked, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "Zone 1 Power (Downtown)", data: [], backgroundColor: "rgba(59, 130, 246, 0.5)", borderColor: "#3b82f6", fill: true, tension: 0.3, pointRadius: 0 },
          { label: "Zone 2 Power (Industrial)", data: [], backgroundColor: "rgba(245, 158, 11, 0.5)", borderColor: "#f59e0b", fill: true, tension: 0.3, pointRadius: 0 },
          { label: "Zone 3 Power (Residential)", data: [], backgroundColor: "rgba(139, 92, 246, 0.5)", borderColor: "#8b5cf6", fill: true, tension: 0.3, pointRadius: 0 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#6b7280", maxTicksLimit: 10 } },
          y: { stacked: true, grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } }
        },
        plugins: { legend: { labels: { color: "#9ca3af" } } }
      }
    });
  }
}

// Load Initial 144 Points of Rich Power Trajectory
async function loadInitialPowerTrajectory() {
  try {
    const res = await fetch(`${API_BASE}/api/power/initial-trajectory?model_name=${state.currentModel}&points=144`);
    if (!res.ok) return;
    const json = await res.json();
    if (!json.data || json.data.length === 0) return;

    const chart = state.charts.liveStream;
    if (chart) {
      chart.data.labels = json.data.map(d => d.time_label);
      chart.data.datasets[0].data = json.data.map(d => d.actual_total);
      chart.data.datasets[1].data = json.data.map(d => d.predicted_total);
      chart.data.datasets[2].data = json.data.map(d => d.actual_zone_1);
      chart.data.datasets[3].data = json.data.map(d => d.actual_zone_2);
      chart.data.datasets[4].data = json.data.map(d => d.actual_zone_3);
      chart.update();
    }

    // Update KPI values from latest trajectory point
    const lastPoint = json.data[json.data.length - 1];
    updateKpiNumbers(lastPoint.actual_total, lastPoint.actual_zone_1, lastPoint.actual_zone_2, lastPoint.actual_zone_3, lastPoint.temperature, lastPoint.humidity, lastPoint.wind_speed, lastPoint.solar_flux);
  } catch (err) {
    console.warn("Failed to load initial trajectory:", err);
  }
}

function updateKpiNumbers(total, z1, z2, z3, temp, hum, wind, solar) {
  const totEl = document.getElementById("kpi-total-val");
  const z1El = document.getElementById("kpi-zone1-val");
  const z2El = document.getElementById("kpi-zone2-val");
  const z3El = document.getElementById("kpi-zone3-val");
  const tempEl = document.getElementById("kpi-temp-val");
  const humEl = document.getElementById("kpi-hum-val");
  const windEl = document.getElementById("kpi-wind-val");
  const solarEl = document.getElementById("kpi-solar-val");

  if (totEl) totEl.innerText = Math.round(total).toLocaleString();
  if (z1El) z1El.innerText = Math.round(z1).toLocaleString();
  if (z2El) z2El.innerText = Math.round(z2).toLocaleString();
  if (z3El) z3El.innerText = Math.round(z3).toLocaleString();
  if (tempEl) tempEl.innerText = `${temp.toFixed(1)}°C`;
  if (humEl) humEl.innerText = `${Math.round(hum)}%`;
  if (windEl) windEl.innerText = `${wind.toFixed(1)} m/s`;
  if (solarEl) solarEl.innerText = `${Math.round(solar)} W/m²`;

  const z1Cur = document.getElementById("zone1-cur-kw");
  const z2Cur = document.getElementById("zone2-cur-kw");
  const z3Cur = document.getElementById("zone3-cur-kw");
  if (z1Cur) z1Cur.innerText = `${Math.round(z1).toLocaleString()} KW`;
  if (z2Cur) z2Cur.innerText = `${Math.round(z2).toLocaleString()} KW`;
  if (z3Cur) z3Cur.innerText = `${Math.round(z3).toLocaleString()} KW`;

  const p1 = document.getElementById("zone1-progress");
  const p2 = document.getElementById("zone2-progress");
  const p3 = document.getElementById("zone3-progress");
  const t1 = document.getElementById("zone1-pct-tag");
  const t2 = document.getElementById("zone2-pct-tag");
  const t3 = document.getElementById("zone3-pct-tag");

  const pct1 = Math.min(100, Math.round((z1 / 52204) * 100));
  const pct2 = Math.min(100, Math.round((z2 / 37409) * 100));
  const pct3 = Math.min(100, Math.round((z3 / 47598) * 100));

  if (p1) p1.style.width = `${pct1}%`;
  if (p2) p2.style.width = `${pct2}%`;
  if (p3) p3.style.width = `${pct3}%`;
  if (t1) t1.innerText = `${pct1}% Capacity`;
  if (t2) t2.innerText = `${pct2}% Capacity`;
  if (t3) t3.innerText = `${pct3}% Capacity`;
}

// Live Telemetry Streaming
function toggleStreaming() {
  state.isStreaming = !state.isStreaming;
  const btn = document.getElementById("toggle-stream-btn");
  if (state.isStreaming) {
    if (btn) {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> Pause Stream`;
      btn.className = "btn-secondary";
    }
    state.streamInterval = setInterval(fetchStreamStep, state.streamSpeedMs);
  } else {
    if (btn) {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Live Telemetry Stream`;
      btn.className = "btn-primary";
    }
    clearInterval(state.streamInterval);
  }
}

async function fetchStreamStep() {
  try {
    const res = await fetch(`${API_BASE}/api/stream/next?model_name=${state.currentModel}`);
    if (!res.ok) return;
    const packet = await res.json();
    
    updateKpiNumbers(packet.actual.total, packet.actual.zone_1, packet.actual.zone_2, packet.actual.zone_3, packet.weather.temperature, packet.weather.humidity, packet.weather.wind_speed, packet.weather.solar_radiation);
    
    const chart = state.charts.liveStream;
    if (chart) {
      const timeLabel = packet.timestamp.split(" ")[1] || packet.timestamp;
      chart.data.labels.push(timeLabel);
      chart.data.datasets[0].data.push(packet.actual.total);
      chart.data.datasets[1].data.push(packet.predicted.total);
      chart.data.datasets[2].data.push(packet.actual.zone_1);
      chart.data.datasets[3].data.push(packet.actual.zone_2);
      chart.data.datasets[4].data.push(packet.actual.zone_3);

      if (chart.data.labels.length > 72) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(ds => ds.data.shift());
      }
      chart.update("none");
    }
  } catch (err) {
    console.error("Stream step error:", err);
  }
}

// Deep Learning Power Predictor (Calculator)
function setupCalculatorListeners() {
  const sliders = [
    { id: "calc-temp", unit: "°C", valId: "calc-temp-val" },
    { id: "calc-hum", unit: "%", valId: "calc-hum-val" },
    { id: "calc-wind", unit: " m/s", valId: "calc-wind-val" },
    { id: "calc-solar", unit: " W/m²", valId: "calc-solar-val" },
    { id: "calc-hour", unit: ":00", valId: "calc-hour-val" },
    { id: "calc-month", unit: "", valId: "calc-month-val" }
  ];

  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  sliders.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) {
      el.addEventListener("input", (e) => {
        const valSpan = document.getElementById(s.valId);
        if (valSpan) {
          if (s.id === "calc-month") {
            valSpan.innerText = `${months[parseInt(e.target.value) - 1]} (Month ${e.target.value})`;
          } else if (s.id === "calc-hour") {
            const h = parseInt(e.target.value);
            const period = h >= 12 && h < 18 ? "Afternoon Peak" : (h >= 18 && h < 23 ? "Evening Domestic Peak" : (h < 6 ? "Night Off-Peak" : "Morning Ramp"));
            valSpan.innerText = `${h}:00 (${period})`;
          } else {
            valSpan.innerText = `${e.target.value}${s.unit}`;
          }
        }
        debounce(triggerCalculatorPrediction, 120)();
      });
    }
  });

  const calcBtn = document.getElementById("calc-predict-btn");
  if (calcBtn) calcBtn.addEventListener("click", triggerCalculatorPrediction);
}

let calcDebounceTimer = null;
function debounce(func, delay) {
  return function(...args) {
    clearTimeout(calcDebounceTimer);
    calcDebounceTimer = setTimeout(() => func.apply(this, args), delay);
  };
}

async function triggerCalculatorPrediction() {
  const temp = parseFloat(document.getElementById("calc-temp")?.value || 22);
  const hum = parseFloat(document.getElementById("calc-hum")?.value || 65);
  const wind = parseFloat(document.getElementById("calc-wind")?.value || 2.0);
  const solar = parseFloat(document.getElementById("calc-solar")?.value || 250);
  const hour = parseInt(document.getElementById("calc-hour")?.value || 14);
  const month = parseInt(document.getElementById("calc-month")?.value || 7);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        temperature: temp,
        humidity: hum,
        wind_speed: wind,
        general_diffuse_flows: solar * 0.7,
        diffuse_flows: solar * 0.3,
        hour: hour,
        month: month,
        day_of_week: 2,
        model_name: state.currentModel
      })
    });
    if (!res.ok) return;
    const pred = await res.json();
    renderCalculatorResults(pred);
  } catch (err) {
    console.error("Calculator prediction error:", err);
  }
}

function renderCalculatorResults(pred) {
  const totKwEl = document.getElementById("calc-res-total-kw");
  const totMwEl = document.getElementById("calc-res-total-mw");
  const z1El = document.getElementById("calc-res-z1-kw");
  const z2El = document.getElementById("calc-res-z2-kw");
  const z3El = document.getElementById("calc-res-z3-kw");
  const riskEl = document.getElementById("calc-res-risk");
  const modelEl = document.getElementById("calc-res-model");
  const latEl = document.getElementById("calc-res-latency");

  if (totKwEl) totKwEl.innerText = `${Math.round(pred.total_power_kw).toLocaleString()} KW`;
  if (totMwEl) totMwEl.innerText = `(${(pred.total_power_kw / 1000).toFixed(2)} MW)`;
  if (z1El) z1El.innerText = `${Math.round(pred.zone_1_kw).toLocaleString()} KW`;
  if (z2El) z2El.innerText = `${Math.round(pred.zone_2_kw).toLocaleString()} KW`;
  if (z3El) z3El.innerText = `${Math.round(pred.zone_3_kw).toLocaleString()} KW`;
  if (modelEl) modelEl.innerText = pred.model_name;
  if (latEl) latEl.innerText = `${pred.latency_ms.toFixed(3)} ms`;

  const bar1 = document.getElementById("calc-bar-z1");
  const bar2 = document.getElementById("calc-bar-z2");
  const bar3 = document.getElementById("calc-bar-z3");
  if (bar1) bar1.style.width = `${Math.round((pred.zone_1_kw / pred.total_power_kw) * 100)}%`;
  if (bar2) bar2.style.width = `${Math.round((pred.zone_2_kw / pred.total_power_kw) * 100)}%`;
  if (bar3) bar3.style.width = `${Math.round((pred.zone_3_kw / pred.total_power_kw) * 100)}%`;

  if (riskEl) {
    riskEl.innerText = `GRID STATUS: ${pred.grid_status}`;
    if (pred.grid_status.includes("CRITICAL")) {
      riskEl.className = "badge-tag badge-rose";
    } else if (pred.grid_status.includes("WARNING")) {
      riskEl.className = "badge-tag badge-amber";
    } else {
      riskEl.className = "badge-tag badge-green";
    }
  }
}

// Dataset Raw Records Viewer
async function loadDatasetRecords(offset) {
  try {
    const res = await fetch(`${API_BASE}/api/power/dataset-records?limit=${state.datasetLimit}&offset=${offset}`);
    if (!res.ok) return;
    const json = await res.json();
    const tbody = document.getElementById("dataset-table-body");
    const info = document.getElementById("dataset-page-info");
    
    if (info) info.innerText = `Rows ${offset + 1}-${Math.min(offset + state.datasetLimit, json.total_rows).toLocaleString()} of ${json.total_rows.toLocaleString()}`;
    if (!tbody || !json.records) return;

    tbody.innerHTML = "";
    json.records.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${r.DateTime}</strong></td>
        <td>${r.Temperature}°C</td>
        <td>${r.Humidity}%</td>
        <td>${r.Wind_Speed} m/s</td>
        <td>${r.General_Diffuse + r.Diffuse} W/m²</td>
        <td style="color:#60a5fa;font-weight:600;">${Math.round(r.Zone_1_KW).toLocaleString()} KW</td>
        <td style="color:#fbbf24;font-weight:600;">${Math.round(r.Zone_2_KW).toLocaleString()} KW</td>
        <td style="color:#c084fc;font-weight:600;">${Math.round(r.Zone_3_KW).toLocaleString()} KW</td>
        <td style="color:#34d399;font-weight:700;">${Math.round(r.Total_Power_KW).toLocaleString()} KW</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Dataset records load error:", err);
  }
}

// Multi Horizon Forecasting Handler
async function triggerForecastUpdate() {
  const horizonEl = document.getElementById("forecast-horizon");
  const horizonHours = horizonEl ? parseInt(horizonEl.value) : 24;

  try {
    const res = await fetch(`${API_BASE}/api/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        horizon_hours: horizonHours,
        model_name: state.currentModel,
        start_offset: 140
      })
    });
    if (!res.ok) return;
    const data = await res.json();
    renderForecastData(data);
  } catch (err) {
    console.error("Forecast update error:", err);
  }
}

function renderForecastData(data) {
  const chart = state.charts.forecast;
  const zoneChart = state.charts.zoneStacked;
  if (!chart || !data.forecast) return;

  const labels = data.forecast.map(f => f.timestamp.split(" ")[1] || f.timestamp);
  const actualTot = data.forecast.map(f => f.actual_total);
  const predTot = data.forecast.map(f => f.predicted_total);
  const upperTot = data.forecast.map(f => f.upper_bound_total);
  const lowerTot = data.forecast.map(f => f.lower_bound_total);

  chart.data.labels = labels;
  chart.data.datasets[0].data = actualTot;
  chart.data.datasets[1].data = predTot;
  chart.data.datasets[2].data = upperTot;
  chart.data.datasets[3].data = lowerTot;
  chart.update();

  if (zoneChart) {
    zoneChart.data.labels = labels;
    zoneChart.data.datasets[0].data = data.forecast.map(f => f.predicted_zone_1);
    zoneChart.data.datasets[1].data = data.forecast.map(f => f.predicted_zone_2);
    zoneChart.data.datasets[2].data = data.forecast.map(f => f.predicted_zone_3);
    zoneChart.update();
  }
}

// Diurnal Curves & Monthly Profile
async function loadHourlyPatterns() {
  try {
    const res = await fetch(`${API_BASE}/api/analytics/hourly-patterns`);
    if (!res.ok) return;
    const patterns = await res.json();
    const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const hData = patterns.hourly_averages;

    const ctxHourly = document.getElementById("chart-hourly-diurnal");
    if (ctxHourly && hData && typeof Chart !== "undefined") {
      new Chart(ctxHourly, {
        type: "line",
        data: {
          labels: hours,
          datasets: [
            { label: "Total City Power Consumption (KW)", data: Object.values(hData["Total Power"]), borderColor: "#10b981", borderWidth: 3.2, tension: 0.35, pointRadius: 2 },
            { label: "Zone 1 Power (Downtown KW)", data: Object.values(hData["Zone 1"]), borderColor: "#3b82f6", borderWidth: 2, tension: 0.35, pointRadius: 2 },
            { label: "Zone 2 Power (Industrial KW)", data: Object.values(hData["Zone 2"]), borderColor: "#f59e0b", borderWidth: 2, tension: 0.35, pointRadius: 2 },
            { label: "Zone 3 Power (Residential KW)", data: Object.values(hData["Zone 3"]), borderColor: "#8b5cf6", borderWidth: 2, tension: 0.35, pointRadius: 2 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#9ca3af" } } },
          scales: {
            x: { grid: { color: "rgba(255, 255, 255, 0.04)" }, ticks: { color: "#9ca3af" } },
            y: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#9ca3af" } }
          }
        }
      });
    }
  } catch (err) {
    console.warn("Failed to load hourly patterns:", err);
  }
}

// Render Benchmark Charts
function renderBenchmarkCharts() {
  if (!state.benchmarks || typeof Chart === "undefined") return;
  const models = Object.keys(state.benchmarks);
  const r2Vals = models.map(m => state.benchmarks[m]["Total Power"].R2);
  const rmseVals = models.map(m => state.benchmarks[m]["Total Power"].RMSE);
  const latVals = models.map(m => state.benchmarks[m].latency_ms_per_sample);

  const ctxBar = document.getElementById("chart-benchmark-bars");
  if (ctxBar) {
    if (state.charts.benchmarkBar) state.charts.benchmarkBar.destroy();
    state.charts.benchmarkBar = new Chart(ctxBar, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Total Power R² Score",
            data: r2Vals,
            backgroundColor: "rgba(6, 182, 212, 0.7)",
            borderColor: "#06b6d4",
            borderWidth: 1,
            yAxisID: "y"
          },
          {
            label: "Total Power RMSE Error (KW)",
            data: rmseVals,
            backgroundColor: "rgba(244, 63, 94, 0.7)",
            borderColor: "#f43f5e",
            borderWidth: 1,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#9ca3af" } } },
        scales: {
          x: { ticks: { color: "#9ca3af" } },
          y: {
            type: "linear",
            display: true,
            position: "left",
            min: 0.95,
            max: 1.0,
            ticks: { color: "#06b6d4" },
            title: { display: true, text: "R² Accuracy Score", color: "#06b6d4" }
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: { color: "#f43f5e" },
            title: { display: true, text: "RMSE Error (KW)", color: "#f43f5e" }
          }
        }
      }
    });
  }

  const ctxRadar = document.getElementById("chart-benchmark-radar");
  if (ctxRadar) {
    if (state.charts.benchmarkRadar) state.charts.benchmarkRadar.destroy();
    const minRmse = Math.min(...rmseVals), maxRmse = Math.max(...rmseVals);
    const minLat = Math.min(...latVals), maxLat = Math.max(...latVals);

    const radarDatasets = models.map((m, idx) => {
      const colors = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];
      const b = state.benchmarks[m];
      const r2Norm = (b["Total Power"].R2) * 100;
      const rmseNorm = (1 - (b["Total Power"].RMSE - minRmse) / (maxRmse - minRmse + 1e-5)) * 100;
      const latNorm = (1 - (b.latency_ms_per_sample - minLat) / (maxLat - minLat + 1e-5)) * 100;
      const paramNorm = (1 - b.parameters / 100000) * 100;

      return {
        label: m,
        data: [r2Norm, rmseNorm, latNorm, paramNorm],
        borderColor: colors[idx % colors.length],
        backgroundColor: colors[idx % colors.length] + "22",
        pointBackgroundColor: colors[idx % colors.length],
        borderWidth: 2
      };
    });

    state.charts.benchmarkRadar = new Chart(ctxRadar, {
      type: "radar",
      data: {
        labels: ["Power R² Fit", "Low RMSE Error", "Inference Velocity", "Model Compactness"],
        datasets: radarDatasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#9ca3af" } } },
        scales: {
          r: {
            angleLines: { color: "rgba(255, 255, 255, 0.08)" },
            grid: { color: "rgba(255, 255, 255, 0.08)" },
            pointLabels: { color: "#cbd5e1" },
            ticks: { display: false }
          }
        }
      }
    });
  }
}
