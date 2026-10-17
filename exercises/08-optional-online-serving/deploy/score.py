"""
Scoring script for the olist-freight-cost model on an Azure ML managed online endpoint.

Azure ML calls:
  init() once, when the container starts  -> load the model into memory
  run()  for every request                -> validate input, predict, return JSON

Request body (one or more order items):
{
  "input_data": [
    {
      "seller_state": "SP", "customer_state": "RJ", "product_category": "health_beauty",
      "price": 59.9, "product_weight_g": 800,
      "product_length_cm": 30, "product_height_cm": 10, "product_width_cm": 20,
      "distance_km": 360
    }
  ]
}
Optional fields (filled in when missing):
  volume_cm3        = length x height x width
  same_state        = 1 if seller_state == customer_state else 0
  items_in_order    = 1
  items_from_seller = 1
  purchase_month    = current month

Response: {"freight_value_brl": [18.42], "model": "olist-freight-cost"}
"""

import json
import logging
import os
from datetime import datetime, timezone

import cloudpickle
import pandas as pd

CATEGORICAL = ["seller_state", "customer_state", "product_category"]
NUMERIC = [
    "price", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    "volume_cm3", "distance_km", "same_state", "items_in_order", "items_from_seller", "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC
REQUIRED = [
    "seller_state", "customer_state", "price", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm", "distance_km",
]

model = None


def _find_model_file(root):
    """The registered model folder is mounted under AZUREML_MODEL_DIR; find model.pkl inside it."""
    for folder, _, files in os.walk(root):
        if "model.pkl" in files:
            return os.path.join(folder, "model.pkl")
    raise FileNotFoundError(f"model.pkl not found under {root}")


def init():
    global model
    path = _find_model_file(os.environ.get("AZUREML_MODEL_DIR", "."))
    with open(path, "rb") as f:
        model = cloudpickle.load(f)
    logging.info("Loaded model from %s", path)


def _prepare(records):
    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED if c not in df.columns or df[c].isna().any()]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    if "product_category" not in df.columns:
        df["product_category"] = "unknown"
    if "volume_cm3" not in df.columns:
        df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    if "same_state" not in df.columns:
        df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    for col in ("items_in_order", "items_from_seller"):
        if col not in df.columns:
            df[col] = 1
    if "purchase_month" not in df.columns:
        df["purchase_month"] = datetime.now(timezone.utc).month

    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    df[NUMERIC] = df[NUMERIC].astype("float64")
    return df[FEATURES]


def run(raw_data):
    try:
        body = json.loads(raw_data)
        records = body.get("input_data", body) if isinstance(body, dict) else body
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records:
            raise ValueError('Send {"input_data": [ {...one order item...}, ... ]}')
        features = _prepare(records)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return {"error": str(error)}

    predictions = model.predict(features)
    return {"freight_value_brl": [round(float(p), 2) for p in predictions], "model": "olist-freight-cost"}
