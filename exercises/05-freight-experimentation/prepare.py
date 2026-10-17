"""Build chronological Olist freight MLTable inputs from the raw Olist CSVs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["seller_state", "customer_state", "product_category"]
NUMERIC = [
    "price", "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "volume_cm3", "distance_km", "same_state",
    "items_in_order", "items_from_seller", "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC


def prepare(raw: Path) -> pd.DataFrame:
    items = pd.read_csv(raw / "olist_order_items_dataset.csv")
    orders = pd.read_csv(raw / "olist_orders_dataset.csv", usecols=[
        "order_id", "customer_id", "order_purchase_timestamp",
    ], parse_dates=["order_purchase_timestamp"])
    customers = pd.read_csv(raw / "olist_customers_dataset.csv", usecols=[
        "customer_id", "customer_zip_code_prefix", "customer_state",
    ], dtype={"customer_zip_code_prefix": "string"})
    sellers = pd.read_csv(raw / "olist_sellers_dataset.csv", usecols=[
        "seller_id", "seller_zip_code_prefix", "seller_state",
    ], dtype={"seller_zip_code_prefix": "string"})
    products = pd.read_csv(raw / "olist_products_dataset.csv")
    translation = pd.read_csv(raw / "product_category_name_translation.csv")
    geo = pd.read_csv(raw / "olist_geolocation_dataset.csv", usecols=[
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
    ], dtype={"geolocation_zip_code_prefix": "string"})

    valid = geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)
    coords = geo.loc[valid].groupby("geolocation_zip_code_prefix")[[
        "geolocation_lat", "geolocation_lng",
    ]].median()

    df = (
        items.merge(orders, on="order_id", validate="many_to_one")
        .merge(customers, on="customer_id", validate="many_to_one")
        .merge(sellers, on="seller_id", validate="many_to_one")
        .merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(translation, on="product_category_name", how="left", validate="many_to_one")
    )
    df = df.join(coords.add_prefix("customer_"), on="customer_zip_code_prefix")
    df = df.join(coords.add_prefix("seller_"), on="seller_zip_code_prefix")
    assert len(df) == len(items) and not df.duplicated(["order_id", "order_item_id"]).any()

    lat1, lon1, lat2, lon2 = map(np.radians, [
        df.seller_geolocation_lat, df.seller_geolocation_lng,
        df.customer_geolocation_lat, df.customer_geolocation_lng,
    ])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 2 * 6371.0088 * np.arcsin(np.sqrt(a.clip(0, 1)))
    df["volume_cm3"] = df.product_length_cm * df.product_height_cm * df.product_width_cm
    df["same_state"] = df.seller_state.eq(df.customer_state).astype(int)
    df["items_in_order"] = df.groupby("order_id").order_item_id.transform("size")
    df["items_from_seller"] = df.groupby(["order_id", "seller_id"]).order_item_id.transform("size")
    df["purchase_month"] = df.order_purchase_timestamp.dt.month
    df["product_category"] = (
        df.product_category_name_english.fillna(df.product_category_name).fillna("unknown")
    )
    assert df.order_purchase_timestamp.notna().all() and df.freight_value.notna().all()
    assert df.freight_value.ge(0).all()
    return df[["order_id", "order_purchase_timestamp", *FEATURES, "freight_value"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True, help="Folder containing the nine Olist CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    df = prepare(args.raw_data)
    windows = {
        "train": df.order_purchase_timestamp.lt("2018-04-01"),
        "validation": df.order_purchase_timestamp.between("2018-04-01", "2018-06-01", inclusive="left"),
        "test": df.order_purchase_timestamp.between("2018-06-01", "2018-09-01", inclusive="left"),
    }
    assert sum(mask.sum() for mask in windows.values()) == len(df) - 1  # One later item is outside the lab window.
    order_sets = [set(df.loc[mask, "order_id"]) for mask in windows.values()]
    assert all(not order_sets[i] & order_sets[j] for i in range(3) for j in range(i + 1, 3))
    for split, mask in windows.items():
        folder = args.output_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        df.loc[mask, [*FEATURES, "freight_value"]].to_csv(folder / "orders.csv", index=False)
        print(f"{split}: {int(mask.sum()):,} items -> {folder / 'orders.csv'}")
    print("Only checkout-time features and the freight target are exported; test is held out.")


if __name__ == "__main__":
    main()
