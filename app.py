import os
import pickle
import pandas as pd
import streamlit as st

# =====================================================================
# 1. PATH & CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "UP_rainfall_dataset.csv")

FEATURE_ORDER = [
    'YEAR', 'MO', 'DY', 'RH2M', 'T2MDEW', 'QV2M', 'PS', 'WS50M', 'T2MWET',
    'WD50M', 'T2M_MAX', 'T2M_MIN', 'ALLSKY_SFC_UV_INDEX', 'TS', 'PSC', 'WSC',
    'LATITUDE', 'LONGITUDE'
]

LOOKUP_COLUMNS = [c for c in FEATURE_ORDER if c not in ("YEAR", "MO", "DY")]

# Cache data loading to prevent reloading on every click
@st.cache_data
def load_historical_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['DISTRICT'] = df['DISTRICT'].str.strip().str.title()
        return df
    return None

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# Load resources
df_historical = load_historical_data()
model = load_prediction_model()

# =====================================================================
# 2. STREAMLIT UI RENDER
# =====================================================================
st.set_page_page_config(page_title="SkySense - Rainfall Predictor", layout="wide")

st.title("⚡ SkySense - Rainfall Prediction Dashboard")
st.markdown("Provide weather observations manually or auto-fill them from historical records.")

# Sidebar for Historical Date Auto-Fill
st.sidebar.header("🔍 Auto-Fill From Historical Date")
st.sidebar.markdown("Pick a date to pull historical averages straight from the dataset.")

af_year = st.sidebar.number_input("Lookup Year", min_value=2005, max_value=2026, value=2026)
af_month = st.sidebar.selectbox("Lookup Month", list(range(1, 13)), index=6) # Default July
af_day = st.sidebar.number_input("Lookup Day", min_value=1, max_value=31, value=15)

# Dictionary to hold default values
default_vals = {
    'RH2M': 88.5, 'T2MDEW': 24.1, 'QV2M': 18.5, 'PS': 99.4, 'T2MWET': 25.3,
    'TS': 26.1, 'T2M_MAX': 32.5, 'T2M_MIN': 24.0, 'WS50M': 5.2, 'WD50M': 145.0,
    'ALLSKY_SFC_UV_INDEX': 0.35, 'PSC': 90.2, 'WSC': 4.8, 'LATITUDE': 27.17, 'LONGITUDE': 78.04
}

if st.sidebar.button("⚡ Load Historical Metrics"):
    if df_historical is not None:
        # Filter data
        exact_match = df_historical[(df_historical["YEAR"] == af_year) & (df_historical["MO"] == af_month) & (df_historical["DY"] == af_day)]
        if not exact_match.empty:
            matches = exact_match
            st.sidebar.success(f"✓ Found exact historical record!")
        else:
            matches = df_historical[(df_historical["MO"] == af_month) & (df_historical["DY"] == af_day)]
            st.sidebar.info(f"✓ Using climatological average across years.")
        
        if not matches.empty:
            averaged = matches[LOOKUP_COLUMNS].mean(numeric_only=True)
            for col in LOOKUP_COLUMNS:
                st.session_state[col] = round(float(averaged[col]), 2)
        else:
            st.sidebar.error("No historical records found for this month/day.")
    else:
        st.sidebar.error("Dataset CSV missing on server.")

# Main Form Infrastructure
with st.form("weather_form"):
    st.subheader("🕒 Temporal Vector")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=int(af_year), key="YEAR_f")
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=int(af_month), key="MO_f")
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=int(af_day), key="DY_f")

    st.subheader("🌡️ Atmospheric & Thermal Diagnostics")
    col4, col5, col6 = st.columns(3)
    with col4:
        rh2m = st.number_input("Relative Humidity at 2M (%)", value=st.session_state.get('RH2M', default_vals['RH2M']))
    with col5:
        t2mdew = st.number_input("Dew Point at 2M (°C)", value=st.session_state.get('T2MDEW', default_vals['T2MDEW']))
    with col6:
        qv2m = st.number_input("Specific Humidity at 2M (g/kg)", value=st.session_state.get('QV2M', default_vals['QV2M']))

    col7, col8, col9 = st.columns(3)
    with col7:
        ps = st.number_input("Surface Pressure (kPa)", value=st.session_state.get('PS', default_vals['PS']))
    with col8:
        t2mwet = st.number_input("Wet Bulb Temp at 2M (°C)", value=st.session_state.get('T2MWET', default_vals['T2MWET']))
    with col9:
        ts = st.number_input("Earth Skin Temperature (°C)", value=st.session_state.get('TS', default_vals['TS']))

    col10, col11 = st.columns(2)
    with col10:
        t2m_max = st.number_input("Max Temp at 2M (°C)", value=st.session_state.get('T2M_MAX', default_vals['T2M_MAX']))
    with col11:
        t2m_min = st.number_input("Min Temp at 2M (°C)", value=st.session_state.get('T2M_MIN', default_vals['T2M_MIN']))

    st.subheader("💨 Aerodynamics & Solar Vectors")
    col12, col13, col14 = st.columns(3)
    with col12:
        ws50m = st.number_input("Wind Speed at 50M (m/s)", value=st.session_state.get('WS50M', default_vals['WS50M']))
    with col13:
        wd50m = st.number_input("Wind Direction at 50M (°)", value=st.session_state.get('WD50M', default_vals['WD50M']))
    with col14:
        uv = st.number_input("UV Index", value=st.session_state.get('ALLSKY_SFC_UV_INDEX', default_vals['ALLSKY_SFC_UV_INDEX']))

    col15, col16 = st.columns(2)
    with col15:
        psc = st.number_input("Corrected Pressure (kPa)", value=st.session_state.get('PSC', default_vals['PSC']))
    with col16:
        wsc = st.number_input("Corrected Wind Speed (m/s)", value=st.session_state.get('WSC', default_vals['WSC']))

    st.subheader("📍 Geospatial Target Coordinates")
    col17, col18 = st.columns(2)
    with col17:
        lat = st.number_input("Latitude", value=st.session_state.get('LATITUDE', default_vals['LATITUDE']))
    with col18:
        lon = st.number_input("Longitude", value=st.session_state.get('LONGITUDE', default_vals['LONGITUDE']))

    submit = st.form_submit_button("Execute Classification Analysis")

# =====================================================================
# 3. PREDICTION PIPELINE CORING
# =====================================================================
if submit:
    if model is None:
        st.error("Error: Trained model file (`trained_model.pkl`) not detected in `model/` folder.")
    else:
        # Build the exact dataframe dictionary structure expected by the pipeline
        features = {
            'YEAR': [float(year)], 'MO': [float(month)], 'DY': [float(day)], 'RH2M': [float(rh2m)],
            'T2MDEW': [float(t2mdew)], 'QV2M': [float(qv2m)], 'PS': [float(ps)], 'WS50M': [float(ws50m)],
            'T2MWET': [float(t2mwet)], 'WD50M': [float(wd50m)], 'T2M_MAX': [float(t2m_max)], 'T2M_MIN': [float(t2m_min)],
            'ALLSKY_SFC_UV_INDEX': [float(uv)], 'TS': [float(ts)], 'PSC': [float(psc)], 'WSC': [float(wsc)],
            'LATITUDE': [float(lat)], 'LONGITUDE': [float(lon)]
        }
        
        input_df = pd.DataFrame(features)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction]) * 100

        st.subheader("📊 Evaluation Report")
        if prediction == 1:
            st.warning(f"🌧️ Precipitation Confirmed (Confidence: {confidence:.2f}%)")
            st.write("The micro-climate matrix indicates atmospheric instability. Rainfall is highly probable.")
        else:
            st.success(f"☀️ Dry / Clear Conditions (Confidence: {confidence:.2f}%)")
            st.write("Atmospheric barometric systems look stable. No precipitation detected for these coordinates.")