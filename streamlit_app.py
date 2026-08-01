"""
Ghana Housing Price Predictor — Streamlit Demo
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Ghana Housing Price Predictor", page_icon="🏠", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ghana_housing_model.pkl")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp { background-color: #FAF6EF; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1C1B1A;
}
section[data-testid="stSidebar"] * {
    color: #F4EEE2 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-weight: 500;
    font-size: 0.85rem;
    color: #C9B79C !important;
}

/* Headline */
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: #1C1B1A;
    margin-bottom: 0.1rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    color: #6B6459;
    font-size: 1rem;
    margin-bottom: 1.4rem;
}

/* Metric cards row */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E0D2;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: left;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8B4A34;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #1C1B1A;
}

/* Result card — the signature element */
.result-card {
    background: #1C1B1A;
    border-radius: 14px;
    padding: 0;
    overflow: hidden;
    margin-top: 0.5rem;
}
.result-stripe {
    height: 6px;
    background: linear-gradient(90deg, #C9962C 0%, #C9962C 48%, #2E5339 52%, #2E5339 100%);
}
.result-inner { padding: 2rem 2.2rem 1.8rem 2.2rem; }
.result-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #C9962C;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-price {
    font-family: 'Fraunces', serif;
    font-size: 3.1rem;
    font-weight: 700;
    color: #FAF6EF;
    line-height: 1.1;
}
.result-range {
    font-size: 0.92rem;
    color: #A69C8A;
    margin-top: 0.6rem;
}

.section-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8B4A34;
    font-weight: 600;
    margin: 1.6rem 0 0.5rem 0;
}

div.stButton > button {
    background-color: #8B4A34;
    color: #FAF6EF;
    border: none;
    font-weight: 600;
    padding: 0.6rem 0;
}
div.stButton > button:hover {
    background-color: #723c2a;
    color: #FAF6EF;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
needs_scaling = bundle["needs_scaling"]
feature_names = bundle["feature_names"]

CITIES = ["Accra", "Kumasi", "Tema", "Takoradi", "Cape Coast", "Tamale"]
PROPERTY_TYPES = ["Apartment", "Detached House", "Townhouse", "Semi-Detached"]


def build_feature_row(payload):
    row = {f: 0 for f in feature_names}
    row["bedrooms"] = payload["bedrooms"]
    row["bathrooms"] = payload["bathrooms"]
    row["size_sqm"] = payload["size_sqm"]
    row["plot_size_sqm"] = payload["plot_size_sqm"]
    row["age_years"] = payload["age_years"]
    row["distance_to_cbd_km"] = payload["distance_to_cbd_km"]
    row["gated_community"] = int(payload["gated_community"])
    row["has_pool"] = int(payload["has_pool"])
    row["has_garage"] = int(payload["has_garage"])
    row["has_backup_power"] = int(payload["has_backup_power"])
    row["has_borehole"] = int(payload["has_borehole"])
    row["price_per_sqm_proxy"] = row["size_sqm"]
    row["total_rooms"] = row["bedrooms"] + row["bathrooms"]
    row["is_far_from_cbd"] = int(row["distance_to_cbd_km"] > 15)

    city = payload["city"]
    if city != "Accra" and f"city_{city}" in row:
        row[f"city_{city}"] = 1

    ptype = payload["property_type"]
    if ptype != "Apartment" and f"property_type_{ptype}" in row:
        row[f"property_type_{ptype}"] = 1

    return pd.DataFrame([row])[feature_names]


# ----------------------------------------------------------------------------
# Sidebar — all inputs live here
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 Property details")

    city = st.selectbox("City", CITIES)
    property_type = st.selectbox("Property type", PROPERTY_TYPES)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    size_sqm = st.slider("Floor size (sqm)", 30, 600, 120)
    plot_size_sqm = st.slider("Plot / land size (sqm) — 0 if apartment", 0, 2000, 0)
    age_years = st.slider("Age of building (years)", 0, 40, 5)
    distance_to_cbd_km = st.slider("Distance to city center (km)", 0.0, 40.0, 10.0)

    st.markdown("---")
    st.markdown("### Amenities")
    gated_community = st.checkbox("Gated community")
    has_pool = st.checkbox("Pool")
    has_garage = st.checkbox("Garage")
    has_backup_power = st.checkbox("Backup power")
    has_borehole = st.checkbox("Borehole")

    st.markdown("---")
    predict_clicked = st.button("Predict price", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Ghana Housing Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Set the property details in the sidebar, then predict an estimated market price.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "This demo is trained on **simulated** data (no live public Ghana housing dataset "
    "was available), so treat predictions as illustrative, not real market advice.",
    icon="⚠️",
)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Model", bundle["model_name"]),
    ("R² (test)", f"{bundle['test_r2']*100:.1f}%"),
    ("Typical error (MAE)", f"GHS {bundle['test_mae']:,.0f}"),
    ("Evaluated on", "held-out test set"),
]
for col, (label, value) in zip([m1, m2, m3, m4], metrics):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Estimate</div>', unsafe_allow_html=True)

if predict_clicked:
    payload = {
        "city": city, "property_type": property_type, "bedrooms": bedrooms,
        "bathrooms": bathrooms, "size_sqm": size_sqm, "plot_size_sqm": plot_size_sqm,
        "age_years": age_years, "distance_to_cbd_km": distance_to_cbd_km,
        "gated_community": gated_community, "has_pool": has_pool,
        "has_garage": has_garage, "has_backup_power": has_backup_power,
        "has_borehole": has_borehole,
    }
    X = build_feature_row(payload)
    X_input = scaler.transform(X) if needs_scaling else X
    pred = max(model.predict(X_input)[0], 80000)
    low, high = pred * 0.87, pred * 1.13  # rough band based on ~13% typical error

    st.markdown(f"""
    <div class="result-card">
        <div class="result-stripe"></div>
        <div class="result-inner">
            <div class="result-label">{bedrooms}-bed {property_type.lower()} · {city}</div>
            <div class="result-price">GHS {pred:,.0f}</div>
            <div class="result-range">Likely range given model error: GHS {low:,.0f} – GHS {high:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="metric-card" style="color:#8B4A34; text-align:center; padding:2rem;">'
        'Fill in the property details in the sidebar, then click <b>Predict price</b>.</div>',
        unsafe_allow_html=True,
    )