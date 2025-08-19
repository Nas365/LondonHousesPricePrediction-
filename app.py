import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, url_for

# Postcode → lat/lon mapping 
POSTCODE_COORDS = {
    "SW1": (51.5018, -0.1416), "SW3": (51.4920, -0.1669),
    "SW6": (51.4759, -0.2060), "SW7": (51.4965, -0.1746),
    "SW8": (51.4782, -0.1369), "SW9": (51.4653, -0.1126),
    "SE1": (51.5050, -0.0850), "SE11": (51.4880, -0.1065),
    "EC1": (51.5246, -0.0985), "WC2": (51.5149, -0.1236)
}
DEFAULT_LON_LAT = (51.5074, -0.1278)  # Central London

def coords_from_postcode_area(outcode: str):
    if not outcode:
        return DEFAULT_LON_LAT
    oc = outcode.strip().upper()
    base = oc if len(oc) <= 3 else oc[:3].rstrip("ABCDEFGHJKLMNOPQRSTUVWXYZ")
    return POSTCODE_COORDS.get(base, DEFAULT_LON_LAT)

# Model download / cache 
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/Nas365/LondonHousesPricePrediction-/releases/download/v1.0/best_random_forest.pkl"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cache.pkl")

def download_model_if_needed():
    """Download the pickle from GitHub Releases if not cached."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL} ...")
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Model download complete.")
    else:
        print(f"Using cached model at {MODEL_PATH}")


def log_transform(x):
    return np.log1p(x)

def load_model():
    download_model_if_needed()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Flask app setup 
app = Flask(__name__)

# Features expected by the model
FEATURES = [
    "latitude", "longitude", "floorAreaSqM",
    "bedrooms", "bathrooms", "livingRooms",
    "propertyType", "tenure",
    "currentEnergyRating", "postcodeArea"
]

# Load model once at startup
model = load_model()

#  Prediction helper 
def model_predict(row_like):
    """Takes a dict or pandas Series, returns prediction."""
    if isinstance(row_like, dict):
        X = pd.DataFrame([row_like], columns=FEATURES)
    elif isinstance(row_like, pd.Series):
        X = pd.DataFrame([row_like.values], columns=FEATURES)
    else:
        X = pd.DataFrame(row_like, columns=FEATURES)
    return float(model.predict(X)[0])

# Health check 
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)})

# Quick HTML form
INDEX_HTML = """
<!doctype html>
<title>London House Price Prediction</title>
<h2>Predict price</h2>
<form method="post" action="/predict-form">
  <p><label>Floor area (sqm) <input name="floorAreaSqM" required step="any" type="number"></label></p>
  <p><label>Bedrooms <input name="bedrooms" required type="number"></label></p>
  <p><label>Bathrooms <input name="bathrooms" required type="number"></label></p>
  <p><label>Living rooms <input name="livingRooms" required type="number"></label></p>
  <p><label>Property type <input name="propertyType" required type="text" placeholder="Flat"></label></p>
  <p><label>Tenure <input name="tenure" required type="text" placeholder="Leasehold"></label></p>
  <p><label>Current energy rating <input name="currentEnergyRating" required type="text" placeholder="D"></label></p>
  <p><label>Postcode area <input name="postcodeArea" required type="text" placeholder="SW1"></label></p>
  <p><button type="submit">Predict</button></p>
</form>
{% if prediction is defined %}
  <h3>Predicted price: £{{ "{:,.0f}".format(prediction) }}</h3>
{% endif %}
"""

# Routes
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.post("/predict-form")
def predict_form():
    data = {k: request.form.get(k) for k in FEATURES if k not in ["latitude", "longitude"]}
    # Auto-fill lat/lon
    lat, lon = coords_from_postcode_area(data.get("postcodeArea", ""))
    data["latitude"] = lat
    data["longitude"] = lon
    # Cast numeric fields
    for k in ["latitude", "longitude", "floorAreaSqM", "bedrooms", "bathrooms", "livingRooms"]:
        v = data.get(k)
        data[k] = float(v) if v not in ("", None) else 0.0
    pred = model_predict(data)
    return render_template_string(INDEX_HTML, prediction=pred)

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    # Build row
    row = {k: payload[k] for k in FEATURES}
    for k in ["latitude", "longitude", "floorAreaSqM", "bedrooms", "bathrooms", "livingRooms"]:
        row[k] = float(row[k])
    pred = model_predict(row)
    return jsonify({"prediction": pred, "currency": "GBP"})

if __name__ == "__main__":
    # Heroku provides the port via env var PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

