"""Prepare a date-corrected Olist late-delivery classification experiment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LAB = Path(__file__).resolve().parent
FEATURES = [
    "customer_state", "seller_state", "promised_days", "distance_km",
    "n_items", "total_price", "total_weight_g", "purchase_month",
]
TARGET = "late_delivery"


def prepare(
    raw: Path, cutoff: str = "2018-06-01", end: str = "2018-09-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(raw / "olist_orders_dataset.csv", usecols=[
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_estimated_delivery_date", "order_delivered_customer_date",
    ], parse_dates=[
        "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date",
    ])
    items = pd.read_csv(raw / "olist_order_items_dataset.csv", usecols=[
        "order_id", "order_item_id", "product_id", "seller_id", "price",
    ])
    products = pd.read_csv(raw / "olist_products_dataset.csv", usecols=["product_id", "product_weight_g"])
    customers = pd.read_csv(raw / "olist_customers_dataset.csv", usecols=[
        "customer_id", "customer_zip_code_prefix", "customer_state",
    ], dtype={"customer_zip_code_prefix": "string"})
    sellers = pd.read_csv(raw / "olist_sellers_dataset.csv", usecols=[
        "seller_id", "seller_zip_code_prefix", "seller_state",
    ], dtype={"seller_zip_code_prefix": "string"})
    geo = pd.read_csv(raw / "olist_geolocation_dataset.csv", usecols=[
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
    ], dtype={"geolocation_zip_code_prefix": "string"})
    assert orders.order_id.is_unique and customers.customer_id.is_unique

    orders = orders[orders.order_status.eq("delivered") & orders.order_delivered_customer_date.notna()].copy()
    orders["promised_days"] = (
        orders.order_estimated_delivery_date - orders.order_purchase_timestamp
    ).dt.total_seconds() / 86400
    # A delivery on the promised calendar date is on time, even if after midnight.
    orders[TARGET] = orders.order_delivered_customer_date.dt.normalize().gt(
        orders.order_estimated_delivery_date.dt.normalize()
    ).astype(int)
    items = items.merge(products, on="product_id", how="left", validate="many_to_one")
    items = items.sort_values(["order_id", "price"], ascending=[True, False])
    baskets = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        total_price=("price", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        seller_id=("seller_id", "first"),
    ).reset_index()
    df = (
        orders.merge(baskets, on="order_id", validate="one_to_one")
        .merge(customers, on="customer_id", validate="many_to_one")
        .merge(sellers, on="seller_id", validate="many_to_one")
    )
    coords = geo.groupby("geolocation_zip_code_prefix")[[
        "geolocation_lat", "geolocation_lng",
    ]].mean()
    df = df.join(coords.add_prefix("customer_"), on="customer_zip_code_prefix")
    df = df.join(coords.add_prefix("seller_"), on="seller_zip_code_prefix")
    lat1, lon1, lat2, lon2 = map(np.radians, [
        df.seller_geolocation_lat, df.seller_geolocation_lng,
        df.customer_geolocation_lat, df.customer_geolocation_lng,
    ])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    df["purchase_month"] = df.order_purchase_timestamp.dt.month
    assert df.order_id.is_unique and df[TARGET].isin([0, 1]).all()
    df = df.sort_values("order_purchase_timestamp")

    train = df[df.order_purchase_timestamp.lt(cutoff)].copy()
    test = df[df.order_purchase_timestamp.between(
        cutoff, end, inclusive="left"
    )].copy()
    assert set(train.order_id).isdisjoint(test.order_id)
    assert len(train) > 0 and len(test) > 0

    # Fill from TRAINING ONLY. RAI components receive the same raw feature columns as the model.
    distance_median = train.distance_km.median()
    for frame in (train, test):
        frame["distance_km"] = frame.distance_km.fillna(distance_median)
        assert frame[FEATURES + [TARGET]].notna().all().all()
    return train[FEATURES + [TARGET]], test[FEATURES + [TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=LAB.parents[1] / "datasets" / "olist" / "Ecommerce Dataset",
                        help="Folder of extracted Olist CSVs (default: shared repository dataset)")
    parser.add_argument("--output-dir", type=Path, default=LAB / "data")
    parser.add_argument("--sweep-only", action="store_true", help="Build pre-April train and April-May validation only")
    args = parser.parse_args()
    if args.sweep_only:
        sweep_train, sweep_validation = prepare(args.raw_data, cutoff="2018-04-01", end="2018-06-01")
        for name, frame in {"sweep-train": sweep_train, "sweep-validation": sweep_validation}.items():
            folder = args.output_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(folder / "data.parquet", index=False)
            print(f"{name}: {len(frame):,} orders; late: {frame[TARGET].sum():,} ({frame[TARGET].mean():.1%})")
        return
    train, test = prepare(args.raw_data)

    # Small, stratified copies bound dashboard run time without changing the full test metrics.
    rai_train, _ = train_test_split(train, train_size=2500, stratify=train[TARGET], random_state=42)
    rai_test, _ = train_test_split(test, train_size=3000, stratify=test[TARGET], random_state=42)
    for name, frame in {
        "train": train, "test": test, "rai-train": rai_train, "rai-test": rai_test,
    }.items():
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / "data.parquet", index=False)
        print(f"{name}: {len(frame):,} orders; late: {frame[TARGET].sum():,} ({frame[TARGET].mean():.1%})")
    print("Only delivered orders with known outcomes are included; this can bias the population.")


if __name__ == "__main__":
    main()
