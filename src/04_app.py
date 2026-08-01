"""
STEP 6 — DEPLOYMENT
A lightweight Flask REST API that loads the trained model and serves
live predictions. Run with: python3 04_app.py
Then POST to http://localhost:5000/predict
"""
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "ghana_housing_model.pkl")

app = Flask(__name__)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
needs_scaling = bundle["needs_scaling"]
feature_names = bundle["feature_names"]

CITIES = ["Accra", "Kumasi", "Tema", "Takoradi", "Cape Coast", "Tamale"]
PROPERTY_TYPES = ["Apartment", "Detached House", "Townhouse", "Semi-Detached"]


def build_feature_row(payload):
    """Turn a plain-English JSON request into the exact feature row the model expects."""
    row = {f: 0 for f in feature_names}

    row["bedrooms"] = payload.get("bedrooms", 3)
    row["bathrooms"] = payload.get("bathrooms", 2)
    row["size_sqm"] = payload.get("size_sqm", 120)
    row["plot_size_sqm"] = payload.get("plot_size_sqm", 0)
    row["age_years"] = payload.get("age_years", 5)
    row["distance_to_cbd_km"] = payload.get("distance_to_cbd_km", 10)
    row["gated_community"] = int(payload.get("gated_community", 0))
    row["has_pool"] = int(payload.get("has_pool", 0))
    row["has_garage"] = int(payload.get("has_garage", 0))
    row["has_backup_power"] = int(payload.get("has_backup_power", 0))
    row["has_borehole"] = int(payload.get("has_borehole", 0))
    row["price_per_sqm_proxy"] = row["size_sqm"]
    row["total_rooms"] = row["bedrooms"] + row["bathrooms"]
    row["is_far_from_cbd"] = int(row["distance_to_cbd_km"] > 15)

    city = payload.get("city", "Accra")
    if city != "Accra" and f"city_{city}" in row:
        row[f"city_{city}"] = 1

    ptype = payload.get("property_type", "Apartment")
    if ptype != "Apartment" and f"property_type_{ptype}" in row:
        row[f"property_type_{ptype}"] = 1

    return pd.DataFrame([row])[feature_names]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": bundle["model_name"], "test_r2": bundle["test_r2"]})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    X = build_feature_row(payload)
    X_input = scaler.transform(X) if needs_scaling else X
    pred = model.predict(X_input)[0]
    pred = max(pred, 80000)  # floor: model shouldn't output below realistic minimum market price
    return jsonify({
        "predicted_price_ghs": round(float(pred), 2),
        "model_used": bundle["model_name"],
        "note": f"Model explains ~{bundle['test_r2']*100:.1f}% of price variation on held-out test data; "
                f"typical error is about GHS {bundle['test_mae']:,.0f}."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
