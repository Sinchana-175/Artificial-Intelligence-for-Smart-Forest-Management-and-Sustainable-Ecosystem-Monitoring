# 🌲 Smart Forest AI: Multi-Sensor Forest Canopy & Wildfire Hazard Monitoring System

An automated, cloud-native environmental monitoring system that combines multi-spectral satellite imagery, synthetic aperture radar (SAR), digital elevation models, and meteorological data to detect forest canopy presence and assess wildfire susceptibility in near-real-time.

---

## 1. Project Overview

Traditional forest monitoring relies heavily on optical satellite data, which fails when:
1. **Deciduous trees shed leaves** during dry summer seasons, lowering optical greenness (NDVI) and causing classifiers to misidentify forests as barren land.
2. **Dense cloud cover or monsoon haze** obscures optical sensors.
3. **Lowland agricultural crops** exhibit high greenness and mimic forest canopies.

**Smart Forest AI** resolves these issues through multi-sensor fusion:
- **Sentinel-2 Optical Data** tracks chlorophyll absorption, canopy greenness, and internal leaf moisture.
- **Sentinel-1 C-Band SAR** penetrates clouds and dry leafless canopies to measure physical volume scattering from tree trunks and woody branches.
- **NASA SRTM Topography** provides elevation and slope context to separate lowland farm tracts from elevated forest reserves.
- **ECMWF ERA5 Climate Data** provides real-time ambient temperature, humidity, and wind vectors to estimate fuel dryness and wildfire spread hazard.

The system extracts spatial features on the fly using **Google Earth Engine (GEE)** and performs inference locally using pre-trained **Scikit-learn** models embedded in an interactive **Streamlit** dashboard.

---

## 2. Technologies & Tools Used

### Languages & Frameworks
- **Python 3.10+**: Core programming language.
- **Streamlit**: Web application framework for the UI and dashboard.
- **Folium / streamlit-folium**: Interactive Leaflet-based geospatial map interface.

### Cloud Computing & Remote Sensing
- **Google Earth Engine (GEE) API (`earthengine-api`)**: Cloud extraction, temporal filtering, and spatial reduction of planetary satellite datasets.
- **Sentinel-2 (MSI Level-2A)**: Multi-spectral optical surface reflectance.
- **Sentinel-1 (SAR C-Band GRD)**: Active microwave radar backscatter ($VV$, $VH$).
- **NASA SRTM (30m GL1)**: Global Digital Elevation Model (DEM) and terrain slope.
- **ECMWF ERA5-Land Reanalysis**: High-resolution hourly atmospheric weather data.
- **MODIS (MCD64A1)**: Historical burned area product (500m thermal burn scars).
- **ESA WorldCover (10m)**: Land use and land cover validation layers.

### Machine Learning & Data Science
- **Scikit-learn**: Offline model training, hyperparameter tuning, and inference pipelines.
- **Pandas & NumPy**: Tabular feature manipulation, array mathematics, and data normalization.
- **Joblib**: Model serialization and loading (`.pkl` binaries).

---

## 3. Machine Learning Models & Evaluated Architectures

The framework evaluates three distinct supervised learning algorithms:

| Model Architecture | 5-Fold Cross Validation | Benchmark Accuracy | Key Structural Advantage |
| :--- | :---: | :---: | :--- |
| **Random Forest (RF)** | **90.80%** | **91.83%** | Ensemble of 100 de-correlated decision trees; highly resilient to non-linear relationships across vastly different sensor scales. |
| **Support Vector Machine (SVM)** | **86.40%** | **85.20%** | Uses an RBF kernel to map complex, non-linear feature interactions into a continuous, maximum-margin separable hyperplane. |
| **Shallow Decision Tree (DT)** | **83.10%** | **82.50%** | Constrained depth baseline providing transparent, rule-based threshold validation. |

---

## 4. End-to-End System Implementation
USER SELECTION (Streamlit Map)
                  [Coordinate Pick via Interactive Folium Map]
                                      │
                                      ▼
                         GOOGLE EARTH ENGINE API
                   (Planetary Data Extraction Pipeline)
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
  [Module 1: Forest Detection]                  [Module 2: Fire Risk Estimation]
   • Sentinel-2 (B3, B4, B8, NDVI, etc.)         • ERA5-Land (Temp, Dew Point, Wind)
   • Sentinel-1 SAR (VV, VH, VH/VV)              • MODIS Fire History (Burned Count)
   • NASA SRTM (Elevation, Slope)                • Live Fuel Indices & Heat-Wind Index
             │                                                 │
             ▼                                                 ▼
    11-Feature Vector Ingestion                   Meteorological Feature Vector Ingestion
             │                                                 │
             ▼                                                 ▼
   Serialized ML Classifier                      Serialized Risk Classifier
    (Random Forest / SVM / DT)                    (Random Forest / SVM / DT)
             │                                                 │
             ▼                                                 ▼
   Primary Forest Identification                 Hazard Tiers: Low / Medium / High
   (Health Status & Cover Change vs 2020)        (Derived from Fuel Aridity & Spread Rate)
             └────────────────────────┬────────────────────────┘
                                      ▼
                           STREAMLIT DASHBOARD
         [Classification Badges, Metric Columns, NDVI Heatmap Tile]

### 4.1 Feature Extraction Workflow (`gee_utils.py`)
1. **Spatial Buffer**: Converts target latitude/longitude into a point geometry with a 10m buffer.
2. **Temporal Compositing**:
   - **Dry Season Sentinel-2**: Composited across `2024-01-01` to `2024-05-31` using `.median()` to filter out cloud contamination and shadow artifacts.
   - **Wet Season Sentinel-2**: Composited across `2024-10-01` to `2024-12-31` to determine peak phenological greening.
   - **Historical Baseline**: Sentinel-2 composite from `2020-01-01` to `2020-05-31` to track long-term canopy changes.
3. **Radar Ingestion**: GEE queries C-band Sentinel-1 GRD imagery in Interferometric Wide (IW) swath mode to extract $VV$ and $VH$ backscatter.
4. **Topography Ingestion**: NASA SRTM queries elevation and computes surface slope angles.
5. **Weather & Fire History Ingestion**: Queries latest ERA5-Land records (temperature, dew point, wind vectors) and sums 5-year historical MODIS burn occurrences.
6. **Unified Reduction**: Combines all rasters into a single multi-band image and runs a single `reduceRegion` call (scale = 10m).

### 4.2 Feature Schema (11 Features for Forest Detection)
- `B3` (Green Band): Surface reflectance at ~560nm.
- `B4` (Red Band): Surface reflectance at ~665nm (chlorophyll absorption).
- `B8` (Near-Infrared Band): Surface reflectance at ~842nm (mesophyll leaf structure).
- `NDVI`: Normalized Difference Vegetation Index: $(B8 - B4) / (B8 + B4)$.
- `NDVI_diff`: Seasonal delta between wet and dry season ($NDVI_{\text{wet}} - NDVI_{\text{dry}}$), capturing deciduous leaf swing.
- `VV`: Co-polarized radar backscatter (surface roughness baseline).
- `VH`: Cross-polarized radar backscatter (volume scattering from tree branches/trunks).
- `VH_VV`: Logarithmic polarization ratio isolating 3D canopy structure from soil moisture.
- `elevation`: Elevation in meters above sea level.
- `slope`: Terrain incline in degrees.
- `landcover`: ESA WorldCover class value.

### 4.3 Inference & Decision Engine (`model_utils.py`)
- **Offline Serialization**: Trained model pipelines are stored locally as `.pkl` binaries inside the `models/` folder.
- **Inference Latency**: GEE completes planetary-scale reduction in ~1–2 seconds. Local model inference on the CPU executes in under 15 milliseconds.
- **Execution Fallbacks**: If serialized model weights encounter missing feature definitions or runtime exceptions, the script automatically deploys rule-based analytical fallbacks to maintain system availability.

---

## 5. System Outputs & Diagnostics

When an analysis is executed, the dashboard outputs:
1. **Primary Status**:
   - `🌲 Forest Canopy Present` (Triggers downstream analytics)
   - `❌ Non-Forest Area Detected` (Immediately halts processing to avoid false environmental reporting)
2. **Vegetation Health**:
   - **Dense**: $NDVI \ge 0.60$
   - **Moderate**: $0.40 \le NDVI < 0.60$
   - **Sparse**: $NDVI < 0.40$
3. **Forest Cover Change Detection (vs. 2020)**:
   - **Deforestation**: $\Delta NDVI \le -0.15$
   - **Afforestation**: $\Delta NDVI \ge +0.15$
   - **No Change**: Stable canopy cover
4. **Wildfire Hazard Level**:
   - Categorized as **Low Risk**, **Medium Risk**, or **High Risk** based on atmospheric temperature, dew point depression, surface wind velocity, and fuel aridity.
5. **Interactive Heatmap Overlay**:
   - Dynamically clips a 2.5 km square region around the target point and renders an encrypted Google Earth Engine tile layer colored across an NDVI color palette (`#d7191c` red to `#1a9641` green).

---

## 6. Project Directory Structure

```text
smart-forest-ai/
│
├── app.py                      # Main Streamlit dashboard application
├── gee_utils.py                # GEE initialization, feature extraction, and tile generation
├── model_utils.py              # ML inference, scaling, fallback logic, and metrics
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
└── models/                     # Serialized model weights (.pkl)
    ├── forest_rf_model.pkl
    ├── forest_svm_model.pkl
    ├── forest_shallow_tree_model.pkl
    ├── fire_rf_model.pkl
    ├── fire_svm_model.pkl
    └── fire_shallow_tree.pkl
    
## 7. Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/smart-forest-ai.git](https://github.com/your-username/smart-forest-ai.git)
cd smart-forest-ai

2. Set Up a Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Authenticate Google Earth Engine
Ensure you have an active Google Cloud project registered for Earth Engine access:
earthengine authenticate
Confirm your Google Cloud project ID in gee_utils.py:
initialize_gee(project_id='your-gee-project-id')

5. Run the Application
streamlit run app.py