#app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import gee_utils
import model_utils

# Page Configuration
st.set_page_config(
    page_title="Smart Forest AI",
    page_icon="🌲",
    layout="wide"
)

# Custom CSS: Stop all UI dimming/fading and enforce crisp borders
st.markdown(
    """
    <style>
    .stApp, div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"], iframe {
        opacity: 1 !important;
        transition: none !important;
        filter: none !important;
    }
    div[data-testid="stBlock"], .element-container, .stMarkdown {
        opacity: 1 !important;
    }
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    iframe {
        border: 1px solid #d0d7de !important;
        border-radius: 0px !important;
    }
    div[data-testid="stCustomComponentV1"], .leaflet-container {
        border-radius: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. State Initialization
if "lat" not in st.session_state:
    st.session_state["lat"] = None
if "lon" not in st.session_state:
    st.session_state["lon"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "heatmap_url" not in st.session_state:
    st.session_state["heatmap_url"] = None

# Initialize Earth Engine and load all paired models
gee_utils.initialize_gee()
all_models = model_utils.load_all_models()

# --- SIDEBAR ---
st.sidebar.title("🌲 Smart Forest AI")
st.sidebar.markdown("### Location & Model Settings")

# Lat/Lon start EMPTY
lat_input = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=st.session_state["lat"],
    placeholder="Select from map or enter lat...",
    format="%.6f",
    step=0.0001
)

lon_input = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=st.session_state["lon"],
    placeholder="Select from map or enter lon...",
    format="%.6f",
    step=0.0001
)

# Synchronize manual typed inputs
if lat_input is not None and lat_input != st.session_state["lat"]:
    st.session_state["lat"] = lat_input
if lon_input is not None and lon_input != st.session_state["lon"]:
    st.session_state["lon"] = lon_input

# Model Selection: Controls BOTH Forest Classifier and Wildfire Estimator
model_name = st.sidebar.radio(
    "Select Model Suite:",
    ["Random Forest", "Support Vector Machine", "Shallow Decision Tree"]
)

run_detection = st.sidebar.button("Run Detection", type="primary", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("🌲 Smart Forest AI Dashboard")
st.markdown("Multi-Sensor Satellite Forest Canopy and Wildfire Hazard Monitoring")

st.markdown("#### Interactive Location Map")
st.caption("Click anywhere on the map to set target coordinates.")

# Default map view center (Bandipur region default)
center_lat = st.session_state["lat"] if st.session_state["lat"] is not None else 11.6664
center_lon = st.session_state["lon"] if st.session_state["lon"] is not None else 76.6291

# Crisp Google Road Map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    attr="Google Maps"
)

# Marker appears on active coordinates
if st.session_state["lat"] is not None and st.session_state["lon"] is not None:
    folium.Marker(
        [st.session_state["lat"], st.session_state["lon"]],
        tooltip=f"Selected: {st.session_state['lat']:.4f}, {st.session_state['lon']:.4f}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

# Dynamic Square Heatmap Overlay
if st.session_state["heatmap_url"]:
    folium.TileLayer(
        tiles=st.session_state["heatmap_url"],
        attr="GEE NDVI",
        name="Canopy Density Heatmap",
        overlay=True,
        control=True,
        opacity=0.75
    ).add_to(m)
    folium.LayerControl().add_to(m)

map_output = st_folium(m, height=520, use_container_width=True, key="interactive_map")

# Sync map click to sidebar coordinates
if map_output and map_output.get("last_clicked"):
    clicked_lat = round(map_output["last_clicked"]["lat"], 6)
    clicked_lon = round(map_output["last_clicked"]["lng"], 6)

    if st.session_state["lat"] is None or abs(clicked_lat - st.session_state["lat"]) > 0.00001 or abs(clicked_lon - st.session_state["lon"]) > 0.00001:
        st.session_state["lat"] = clicked_lat
        st.session_state["lon"] = clicked_lon
        st.session_state["analysis_done"] = False
        st.session_state["heatmap_url"] = None
        st.rerun()

# --- EXECUTION ---
if run_detection:
    if st.session_state["lat"] is None or st.session_state["lon"] is None:
        st.sidebar.warning("⚠️ Please select a location on the map or enter coordinates first.")
    else:
        with st.spinner("Retrieving multi-sensor satellite telemetry..."):
            features = gee_utils.extract_all_satellite_features(st.session_state["lat"], st.session_state["lon"])
            
            # Select specific model weights for BOTH Forest and Fire
            forest_model_obj = all_models["forest"].get(model_name)
            fire_model_obj = all_models["fire"].get(model_name)

            # Forest inference
            is_forest = model_utils.predict_forest_status(features, forest_model_obj, model_name=model_name)

            st.session_state["results"] = {
                "features": features,
                "is_forest": is_forest,
                "model_name": model_name
            }

            if is_forest:
                st.session_state["heatmap_url"] = gee_utils.get_ndvi_heatmap_tile(st.session_state["lat"], st.session_state["lon"])
            else:
                st.session_state["heatmap_url"] = None

            st.session_state["analysis_done"] = True
            st.rerun()

# --- RESULTS DISPLAY ---
if st.session_state["analysis_done"] and st.session_state["results"]:
    res = st.session_state["results"]
    features = res["features"]
    is_forest = res["is_forest"]
    active_model_name = res["model_name"]

    st.markdown("---")
    st.subheader(f"Detection Results: {active_model_name}")

    if not is_forest:
        # Non-forest: immediately stops and suppresses downstream analytics
        st.error("❌ Non-Forest Area Detected")
        st.info("Vegetation health, forest change detection, and wildfire risk analysis are not applicable for non-forest land cover.")
    else:
        st.success("🌲 Forest Canopy Present")

        col1, col2, col3 = st.columns(3)

        # 1. Vegetation Health: Strictly Dense, Moderate, or Sparse
        health_status = model_utils.get_vegetation_health(features.get('NDVI'))
        with col1:
            st.metric(label="Vegetation Health", value=health_status)

        # 2. Forest Change Detection: Strictly Afforestation, Deforestation, or No Change
        change_status = model_utils.get_change_detection(features.get('NDVI'), features.get('NDVI_2020'))
        with col2:
            st.metric(label="Forest Change Detection", value=change_status)

        # 3. Wildfire Hazard Level: Model-specific evaluation
        active_fire_model = all_models["fire"].get(active_model_name)
        fire_risk = model_utils.predict_fire_risk(features, active_fire_model, model_name=active_model_name)
        with col3:
            st.metric(label="Wildfire Hazard Level", value=fire_risk)

        # Model Performance Cards
        st.markdown("#### Model Performance Metrics")
        metrics_info = model_utils.get_model_metrics(active_model_name)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.info(f"**Forest Classification 5-Fold CV:** {metrics_info['cv_acc']}")
        with m_col2:
            st.info(f"**Wildfire Model Benchmark Accuracy:** {metrics_info['test_acc']}")