#gee_utils.py
import ee
import math
import streamlit as st

def initialize_gee(project_id='smart-forest-ai-504713'):
    """Initializes Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
        except Exception as e:
            st.error(f"Earth Engine Initialization Error: {e}")

def _safe_float(val, default=0.0):
    try:
        if val is None or math.isnan(float(val)):
            return default
        return float(val)
    except Exception:
        return default

def _safe_int(val, default=0):
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        return default

@st.cache_data(show_spinner=False)
def extract_all_satellite_features(lat, lon):
    """
    Extracts multi-sensor optical, radar, topographic, weather, and MODIS fire records
    over a 10m buffer in a single consolidated Earth Engine call.
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(10)

    # 1. Sentinel-2 Optical (Dry, Wet, and Historical 2020)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
           .filterBounds(point) \
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    dry_s2 = s2.filterDate('2024-01-01', '2024-05-31').median()
    wet_s2 = s2.filterDate('2024-10-01', '2024-12-31').median()
    hist_s2 = s2.filterDate('2020-01-01', '2020-05-31').median()

    ndvi_dry = dry_s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_wet = wet_s2.normalizedDifference(['B8', 'B4']).rename('NDVI_wet')
    ndvi_hist = hist_s2.normalizedDifference(['B8', 'B4']).rename('NDVI_2020')

    # 2. Sentinel-1 SAR Radar (C-Band)
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
           .filterBounds(point) \
           .filter(ee.Filter.eq('instrumentMode', 'IW')) \
           .filterDate('2024-01-01', '2024-05-31') \
           .median()

    vv = s1.select('VV')
    vh = s1.select('VH')

    # 3. NASA SRTM & ESA WorldCover
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation).select('slope')
    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select('Map').rename('landcover')

    # 4. ECMWF ERA5-Land Weather
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_BY_HOUR") \
             .filterBounds(point) \
             .sort('system:time_start', False) \
             .first()

    temp_c = era5.select('temperature_2m').subtract(273.15).rename('temp_c')
    dew_c = era5.select('dewpoint_temperature_2m').subtract(273.15).rename('dew_c')
    u_wind = era5.select('u_component_of_wind_10m')
    v_wind = era5.select('v_component_of_wind_10m')
    wind_speed = u_wind.hypot(v_wind).rename('wind_speed')

    # 5. MODIS MCD64A1 Historical Burned Count
    modis_fire = ee.ImageCollection("MODIS/061/MCD64A1") \
                   .filterBounds(point) \
                   .filterDate('2019-01-01', '2024-05-31') \
                   .select('BurnDate') \
                   .count() \
                   .rename('burned_count')

    # Combine into single multi-band image for fast reduction
    combined_img = ee.Image.cat([
        ndvi_dry, ndvi_wet, ndvi_hist,
        dry_s2.select(['B3', 'B4', 'B8']),
        vv, vh, elevation, slope, worldcover,
        temp_c, dew_c, wind_speed, modis_fire
    ])

    raw = combined_img.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=roi,
        scale=10,
        maxPixels=1e6
    ).getInfo()

    if raw is None:
        raw = {}

    ndvi = _safe_float(raw.get('NDVI'), 0.45)
    ndvi_wet_val = _safe_float(raw.get('NDVI_wet'), ndvi)
    ndvi_2020 = _safe_float(raw.get('NDVI_2020'), ndvi)
    
    temp = _safe_float(raw.get('temp_c'), 28.0)
    dew = _safe_float(raw.get('dew_c'), 20.0)
    if dew > temp:
        dew = temp - 1.0

    wind = _safe_float(raw.get('wind_speed'), 2.5)
    vh_val = _safe_float(raw.get('VH'), -14.0)
    vv_val = _safe_float(raw.get('VV'), -8.0)
    vh_vv = vh_val / vv_val if vv_val != 0 else 0.5
    burned_count = _safe_int(raw.get('burned_count'), 0)

    # Derived physics features
    temp_dew_spread = temp - dew
    heat_wind_index = (temp / 20.0) * wind
    fuel_aridity = max(0.0, (1.0 - ndvi)) * temp_dew_spread

    return {
        'NDVI': ndvi,
        'NDVI_diff': ndvi_wet_val - ndvi,
        'NDVI_2020': ndvi_2020,
        'B3': _safe_float(raw.get('B3'), 800.0),
        'B4': _safe_float(raw.get('B4'), 700.0),
        'B8': _safe_float(raw.get('B8'), 2800.0),
        'VH': vh_val,
        'VV': vv_val,
        'VH_VV': vh_vv,
        'elevation': _safe_float(raw.get('elevation'), 300.0),
        'slope': _safe_float(raw.get('slope'), 5.0),
        'landcover': _safe_int(raw.get('landcover'), 10),
        'temp_c': temp,
        'dew_c': dew,
        'wind_speed': wind,
        'burned_count': burned_count,
        'temp_dew_spread': temp_dew_spread,
        'heat_wind_index': heat_wind_index,
        'fuel_aridity': fuel_aridity
    }

def get_ndvi_heatmap_tile(lat, lon):
    """Generates an encrypted GEE tile URL with a strict SQUARE bounding box."""
    try:
        point = ee.Geometry.Point([lon, lat])
        square_area = point.buffer(2500).bounds()

        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
               .filterBounds(square_area) \
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
               .filterDate('2024-01-01', '2024-05-31') \
               .median()

        ndvi = s2.normalizedDifference(['B8', 'B4'])
        vis_params = {
            'min': 0.1, 
            'max': 0.8, 
            'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
        }
        map_id = ndvi.clip(square_area).getMapId(vis_params)
        return map_id['tile_fetcher'].url_format
    except Exception:
        return None