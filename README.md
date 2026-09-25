# ♻️ EcoPriority AI — Smart Waste Collection Priority Predictor

> **PS 17 |** AI-powered command center that predicts **next-period bin fill level** and **overflow risk**, then auto-generates a prioritized dispatch queue for Mathura city fleet operations.

Built with **scikit-learn (Dual Random Forest)** + **Streamlit + Plotly**. Entered / built for hackathon-style problem statement **PS-17 Smart Waste Collection Priority Predictor**.

---

## 📌 Problem Statement

Municipal waste collection is still mostly fixed-schedule: trucks visit all bins whether full or empty. This wastes fuel, misses overflowing bins, and creates unhygienic hotspots near high footfall areas.

**EcoPriority AI solves this by:**
1. Learning fill-rate patterns from sensor telemetry (fill level, lags, time-since-collection, weather, events, activity, zone).
2. Predicting **next fill %** (regression) + **overflow probability** (classification) for every bin.
3. Converting predictions into **URGENT / SOON / LOW** priority tiers with an explainable heuristic.
4. Visualizing it all on a live Mathura city dashboard with map routing, dispatch simulation, and batch CSV prediction.

## ✨ Features

### 🖥️ 1. Command Center Dashboard (`app.py` — ~1700 lines)
Dark glassmorphism UI with custom CSS, Plus Jakarta Sans font, KPI cards, badges, pulsing live status.

**Top KPI strip (live-reactive to filters/uploads):**
- 🔴 Urgent Bins | 🟠 Soon Priority | 🟢 Low / Safe | 📈 Fleet Avg Fill % | ⚠️ High Overflow Risk (>70%)

**Sidebar Command Controls:**
- Upload custom telemetry CSV (refreshes *entire* app — KPIs, map, queue, inspector)
- Mathura location / zone filter, priority filter, bin search, fill-level range slider
- Adjustable heuristic thresholds (Urgent Fill / Risk, Soon Fill / Risk)
- Simulated dispatch tracker + reset
- One-click dispatch exports: `priority_collection_list.csv` + `dispatch_manifest.json`

### 🚨 Tab 1 — Dispatch & Priority Queue
- Sortable, color-coded priority table (Bin ID, Mathura location, current fill, predicted next fill, overflow risk, hours-since-empty)
- Priority distribution donut chart + zone-wise avg fill vs. risk grouped bar chart
- Quick actions: **🚛 Empty Bin** and **⚡ Dispatch Entire Location** (session-state simulation with toast + live re-ranking)

### 🗺️ Tab 2 — Geospatial Route & Urgency Grid
- 5 Mathura zones mapped to real places:
  - `ZONE_A` — 🏛️ Krishna Janmabhoomi
  - `ZONE_B` — 🌊 Vishram Ghat
  - `ZONE_C` — 🎓 GLA University Hub
  - `ZONE_D` — 🚆 Mathura Junction
  - `ZONE_E` — 🛣️ Govardhan Chauraha
- Zone urgency grid cards (urgent/soon/safe, avg fill, bin counts)
- Interactive `plotly.scatter_map` with **OpenStreetMap / Carto-Darkmatter toggle**, sized by fill level, colored by priority, landmark ⭐ pins, and auto-drawn **urgent dispatch route polyline**
- Suggested route sequence (top 8 urgent bins by overflow risk) + estimated waste load in tonnes

### 🔍 Tab 3 — Bin Telemetry Deep Dive
- Per-bin inspector with status card (location, weather/event flags, activity index, hours-since-collection)
- Plotly gauge dials for predicted next fill and overflow risk
- Full telemetry history line chart with overflow `X` markers + 90% urgent threshold line (graceful fallback snapshot card for single-row CSVs)
- Mark-as-collected button per bin

### 🧪 Tab 4 — What-If AI Simulation Studio
- Sliders/selectors for fill level, lags, hours-since-collection, activity, weather, event, day-type, zone
- **Live inference** on both Random Forest models with delta display, priority badge, heuristic rationale, and current-vs-predicted bar chart

### 📈 Tab 5 — Model Performance Lab
- Evaluated on time-based 80/20 per-bin holdout
- Metrics: Regression **MAE / RMSE / R²**, Classification **F1 / Precision / Recall**
- Predicted-vs-actual scatter, confusion matrix heatmap, Random Forest feature-importance bar

### 📁 Tab 6 — Files & Pipeline Hub
- Visual architecture flow: `dataset.csv → ps17_pipeline.py → .joblib models → Plots/ + priority list → app.py`
- File cards with live sizes, raw sensor data inspector (100-row preview), pipeline code snippet viewer, saved-plots gallery

### 📤 Tab 7 — Custom CSV Batch Prediction
- Drag-and-drop any CSV — `prepare_features_for_inference()` auto-adapts raw or pre-engineered schemas (parses dates, derives lags/rolling means, normalizes Mathura place names via fuzzy `map_to_zone_code()`, one-hot encodes zones, fills sensible defaults)
- Batch KPI cards, priority pie + fill-vs-risk scatter with threshold lines, full predictions table, CSV/JSON download
- Revert-to-default fleet button; also ships a downloadable sample Mathura template

> No `waste_collection_dataset.csv`? No problem — `generate_demo_dataset()` synthesizes a realistic 100-bin / 2000-row telemetry fleet on the fly.

---

## 🧠 ML Pipeline (`ps17_pipeline.py`)

End-to-end 8-step pipeline: `python ps17_pipeline.py`

| Step | What happens |
|------|--------------|
| 1. Load | Parse `timestamp`, `last_collection`, sort by `bin_id` + time |
| 2. EDA | Distributions, boxplots by zone/day-type, sample-bin time series, correlation heatmap → `Plots/` (mirrored in `EDA_images/`) |
| 3. Feature Engineering | `hours_since_collection`, `hour_of_day`, `day_of_week`, `fill_lag_1`, `fill_lag_2`, `fill_diff_1`, `fill_roll_mean_3`, `day_type_enc`, weather/event/activity, `zone_*` one-hots; targets `target_next_fill` / `target_next_overflow` |
| 4. Split | **Time-based 80/20 per bin** (no shuffle leakage) |
| 5. Regression | Baseline `LinearRegression` vs main `RandomForestRegressor(n_estimators=300, max_depth=10)` → MAE/RMSE/R² + pred-vs-actual + importance plots |
| 6. Classification | Baseline `LogisticRegression` vs main `RandomForestClassifier(n_estimators=300, max_depth=10)` → F1/Precision/Recall + report + confusion matrix |
| 7. Priority List | Latest reading per bin → predict → heuristic labeling → `priority_collection_list.csv` |
| 8. Save | `fill_level_regressor.joblib` + `overflow_classifier.joblib` (Git LFS) |

**Priority heuristic (customizable in sidebar):**
```
URGENT if predicted_next_fill >= 90%  OR overflow_risk >= 0.70
SOON   if predicted_next_fill >= 70%  OR overflow_risk >= 0.40
LOW    otherwise
```

**Feature set (16 cols):** `hours_since_collection, hour_of_day, day_of_week, fill_lag_1, fill_lag_2, fill_diff_1, fill_roll_mean_3, day_type_enc, weather_flag, event_flag, nearby_activity, zone_ZONE_A/B/C/D/E`

---

## 📂 Project Structure

```
├── app.py                        # Streamlit command center (7 tabs)
├── ps17_pipeline.py              # EDA + features + training + evaluation + priority list
├── fill_level_regressor.joblib   # Trained RandomForestRegressor (Git LFS)
├── overflow_classifier.joblib    # Trained RandomForestClassifier (Git LFS)
├── EDA_images/                   # 8 saved EDA & evaluation charts (01–08)
├── .streamlit/config.toml        # Green-on-dark theme + server config
├── .devcontainer/                # Dev container setup
└── README.md
```

> Note: `*.csv`, `Plots/`, `plots/` are gitignored. The app works without the raw CSV via demo-data fallback. Run the pipeline locally to regenerate `Plots/` + `priority_collection_list.csv`.

**Expected dataset schema (`waste_collection_dataset.csv`):**
`bin_id, timestamp, fill_level, location_zone, last_collection, day_type, weather_flag, event_flag, nearby_activity, recent_fill_trend, overflow`

## 🖼️ EDA & Evaluation Glimpses

All charts live in `EDA_images/`:

| # | File | Insight |
|---|------|---------|
| 01 | `01_fill_level_distribution.png` | Overall fill % distribution + KDE |
| 02 | `02_fill_level_by_zone.png` | Zone-wise fill spread (Mathura areas) |
| 03 | `03_fill_level_by_daytype.png` | Weekday vs weekend load |
| 04 | `04_sample_bin_timeseries.png` | Fill-and-reset collection cycles |
| 05 | `05_correlation_heatmap.png` | Numeric feature correlations |
| 06 | `06_regression_pred_vs_actual.png` | RF regressor calibration |
| 07 | `07_regression_feature_importance.png` | Top drivers (lags, hours-since-collection, activity) |
| 08 | `08_confusion_matrix.png` | Overflow classifier errors |

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.10+ recommended.

### 2. Install
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib matplotlib seaborn
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. (Optional) Retrain everything
```bash
# place waste_collection_dataset.csv next to the script, then:
python ps17_pipeline.py
```
This regenerates `Plots/`, `priority_collection_list.csv`, and both `.joblib` models.

### 5. Try custom batch prediction
Open **Tab 7 → Download Mathura Sample CSV → Upload it back** (or upload your own CSV with at least `bin_id` + fill info — missing columns are auto-filled).

---

## 🛠️ Tech Stack
- **ML:** scikit-learn (RandomForest ×2, Linear/Logistic baselines), StandardScaler, time-based split
- **App:** Streamlit, Plotly (Express + Graph Objects + scatter_map), pandas, numpy, joblib
- **Training/EDA:** matplotlib, seaborn
- **Infra:** Git LFS for models, Streamlit theme config, devcontainer-ready

## 🔮 Future Scope
- Route optimization (TSP / distance-aware sequencing instead of risk-sorted order)
- Live IoT ingestion (MQTT / API polling) + alerts (SMS/WhatsApp for URGENT bins)
- Fill-rate forecasting horizon (24h/48h) + truck capacity-constrained dispatch
- Map upgrade: real road-network routing + bin-level lat/lon from GPS
- Model registry, drift monitoring, and lighter models (LightGBM / quantized) for edge deployment

---

♻️ **EcoPriority AI (PS 17)** — *Predict overflow before it happens. Dispatch smarter, not harder.*
