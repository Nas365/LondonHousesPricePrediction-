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
    import sys, types
    if "__main__" not in sys.modules:
        sys.modules["__main__"] = types.SimpleNamespace()
    setattr(sys.modules["__main__"], "log_transform", log_transform)
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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>London House Price Prediction</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
  /* Full-bleed background image (no dark overlay) */
  body {
    min-height: 100vh;
    margin: 0;
    background: url("{{ url_for('static', filename='img/london.jpg') }}") center / cover no-repeat fixed;
    font-family: Arial, sans-serif;
  }

  /* White card */
  .card {
    background: rgba(255, 255, 255, 0.96);   /* clean white */
    border: none;
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.28);
  }

  /* Title + text colors */
  .brand {
    font-weight: 700;
    font-size: 1.8rem;
    color: #111;            /* dark text on white card */
  }
  .lead, label, .form-label {
    color: #222;
  }

  /* Inputs */
  .form-control, .form-select {
    background: #fff;
    color: #111;
    border: 1px solid #d0d5dd;
  }
  .form-control::placeholder { color: #9aa3af; }

  /* Button */
  .btn-primary {
    background-color: #0d6efd;
    border-color: #0d6efd;
  }

  /* Footer note inside the card */
  .footer {
    color: #555;
    font-size: 0.9rem;
    opacity: 0.95;
  }
</style>

</head>
<div class="container py-5 d-flex justify-content-start">
  <div class="col-lg-5">
    <div class="card p-4">
          <h1 class="brand mb-3">🏙️ London House Price Prediction</h1>
          <p class="mb-4">Enter property details below to estimate the sale price.</p>

          <form method="post" action="/predict-form" class="row g-3">
            <div class="col-6">
              <label class="form-label">Floor area (sqm)</label>
              <input class="form-control" type="number" step="any" name="floorAreaSQM" required>
            </div>
            <div class="col-6">
              <label class="form-label">Bedrooms</label>
              <input class="form-control" type="number" name="bedrooms" required>
            </div>
            <div class="col-6">
              <label class="form-label">Bathrooms</label>
              <input class="form-control" type="number" name="bathrooms" required>
            </div>
            <div class="col-6">
              <label class="form-label">Living rooms</label>
              <input class="form-control" type="number" name="livingRooms" required>
            </div>
            <div class="col-6">
              <label class="form-label">Property type</label>
              <input class="form-control" type="text" name="propertyType" placeholder="Flat / Terraced / Detached" required>
            </div>
            <div class="col-6">
              <label class="form-label">Tenure</label>
              <input class="form-control" type="text" name="tenure" placeholder="Leasehold / Freehold" required>
            </div>
            <div class="col-6">
              <label class="form-label">Energy rating</label>
              <input class="form-control" type="text" name="currentEnergyRating" placeholder="A–G" required>
            </div>
            <div class="col-6">
              <label class="form-label">Postcode area</label>
              <input class="form-control" type="text" name="postcodeArea" placeholder="E1 / SW1 / NW3…" required>
            </div>

            <div class="col-12 d-grid">
              <button class="btn btn-primary btn-lg" type="submit">Predict price</button>
            </div>
          </form>

          {% if prediction is defined %}
          <hr class="border-light my-4">
          <h3 class="mb-0">Predicted price: £{{ '{:,.0f}'.format(prediction) }}</h3>
          {% endif %}

          <div class="footer mt-4">Model status: {{ '✅ cached' if model_loaded else '⬇️ downloading' }}</div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
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

