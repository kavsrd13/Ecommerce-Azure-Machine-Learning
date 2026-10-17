"""
STEP 1 of the pipeline - DATA PROCESSING.

Turns the raw Olist tables into ONE ROW PER DELIVERED ORDER with:
  * the target  : delivery_days = time from purchase to delivery, in days
  * features    : only information that is known when the customer places the order

    python prep.py --raw_data "<folder with the Olist CSVs>" --output_data "<output folder>"

In Azure ML, --raw_data is the mounted data asset azureml:olist-ecommerce-raw:1 and
--output_data is a pipeline output folder that the training step reads.
Output: <output_data>/orders_features.csv
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd


def load_csv(folder, name):
    return pd.read_csv(os.path.join(folder, name))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_order_features(raw):
    orders = load_csv(raw, "olist_orders_dataset.csv")
    items = load_csv(raw, "olist_order_items_dataset.csv")
    customers = load_csv(raw, "olist_customers_dataset.csv")
    sellers = load_csv(raw, "olist_sellers_dataset.csv")
    products = load_csv(raw, "olist_products_dataset.csv")
    geolocation = load_csv(raw, "olist_geolocation_dataset.csv")
    translation = load_csv(raw, "product_category_name_translation.csv")

    # --- 1. Keep delivered orders and compute the TARGET ----------------------------------
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col])
    orders = orders[(orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].notna()].copy()
    seconds_per_day = 86400
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / seconds_per_day
    # The date Olist PROMISED at checkout - known in advance, so it is a legal feature
    orders["promised_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / seconds_per_day

    # --- 2. Summarise the items of each order (an order can contain several items) --------
    items = items.merge(products, on="product_id", how="left").merge(translation, on="product_category_name", how="left")
    items["volume_cm3"] = items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    items["product_category"] = items["product_category_name_english"].fillna(items["product_category_name"])
    # The most expensive item decides the "main" seller and category of the order
    items = items.sort_values(["order_id", "price"], ascending=[True, False])
    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        seller_id=("seller_id", "first"),
        product_category=("product_category", "first"),
    ).reset_index()

    df = (
        orders[["order_id", "customer_id", "order_purchase_timestamp", "promised_days", "delivery_days"]]
        .merge(per_order, on="order_id")
        .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_state"]], on="customer_id")
        .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id")
    )

    # --- 3. Distance between seller and customer (average coordinates per zip prefix) -----
    zip_coords = geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
    df = df.join(zip_coords.add_prefix("customer_"), on="customer_zip_code_prefix")
    df = df.join(zip_coords.add_prefix("seller_"), on="seller_zip_code_prefix")
    df["distance_km"] = haversine_km(df["seller_geolocation_lat"], df["seller_geolocation_lng"],
                                     df["customer_geolocation_lat"], df["customer_geolocation_lng"])
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(float)

    # --- 4. Calendar features of the purchase moment ---------------------------------------
    ts = df["order_purchase_timestamp"]
    df["purchase_month"] = ts.dt.month
    df["purchase_weekday"] = ts.dt.weekday   # 0 = Monday
    df["purchase_hour"] = ts.dt.hour
    df["product_category"] = df["product_category"].fillna("unknown")

    columns = [
        "order_id", "order_purchase_timestamp",                      # identification + split
        "seller_state", "customer_state", "product_category",        # categorical features
        "distance_km", "same_state", "promised_days", "n_items", "n_sellers", "total_price",
        "total_freight", "total_weight_g", "total_volume_cm3",
        "purchase_month", "purchase_weekday", "purchase_hour",       # numeric features
        "delivery_days",                                             # target
    ]
    return df[columns].sort_values("order_purchase_timestamp")


def main():
    parser = argparse.ArgumentParser(description="Build the order-level delivery-time dataset.")
    parser.add_argument("--raw_data", required=True, help="Folder with the Olist CSV files")
    parser.add_argument("--output_data", required=True, help="Folder to write orders_features.csv to")
    args = parser.parse_args()

    df = build_order_features(args.raw_data)
    os.makedirs(args.output_data, exist_ok=True)
    out_file = os.path.join(args.output_data, "orders_features.csv")
    df.to_csv(out_file, index=False)

    stats = {
        "orders": len(df),
        "delivery_days_mean": float(df["delivery_days"].mean()),
        "delivery_days_median": float(df["delivery_days"].median()),
        "missing_distance_share": float(df["distance_km"].isna().mean()),
    }
    with mlflow.start_run():
        mlflow.log_metrics(stats)
    print(f"Wrote {len(df):,} orders to {out_file}")
    print(f"delivery_days: mean {stats['delivery_days_mean']:.1f}, median {stats['delivery_days_median']:.1f}")


if __name__ == "__main__":
    main()
