"""
Train a model that predicts the SHIPPING COST (freight_value, in Brazilian reais) of an
Olist order item, using only information that is known when the customer checks out.

The same script runs on your laptop and as an Azure ML job:

    python src/train.py --data "<folder with the 9 Olist CSV files>"

In Azure ML, --data is the mounted data asset  azureml:olist-ecommerce-raw:1.
Parameters, metrics, plots and the trained model are logged with MLflow: to the Azure ML
workspace when running as a job. On a laptop, point MLflow somewhere first, for example:
    $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

Three models are compared on the same test set:
  1. baseline  - always predicts the median freight of the training data
  2. linear    - ridge regression on scaled numeric + one-hot categorical features
  3. boosting  - histogram gradient boosting (the model we keep)
"""

import argparse
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")  # no screen on a cloud VM: draw plots straight to files
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "freight_value"
CATEGORICAL = ["seller_state", "customer_state", "product_category"]
NUMERIC = [
    "price",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "volume_cm3",
    "distance_km",
    "same_state",
    "items_in_order",
    "items_from_seller",
    "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC


# -----------------------------------------------------------------------------------------
# 1. Data -> one row per order item with features known at checkout time
# -----------------------------------------------------------------------------------------
def load_csv(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_dataset(data_dir):
    items = load_csv(data_dir, "olist_order_items_dataset.csv")
    orders = load_csv(data_dir, "olist_orders_dataset.csv")
    customers = load_csv(data_dir, "olist_customers_dataset.csv")
    sellers = load_csv(data_dir, "olist_sellers_dataset.csv")
    products = load_csv(data_dir, "olist_products_dataset.csv")
    geolocation = load_csv(data_dir, "olist_geolocation_dataset.csv")
    translation = load_csv(data_dir, "product_category_name_translation.csv")

    # The geolocation table has many rows per zip-code prefix -> average them.
    zip_coords = geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()

    df = (
        items.merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id")
        .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_state"]], on="customer_id")
        .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id")
        .merge(products, on="product_id", how="left")
        .merge(translation, on="product_category_name", how="left")
    )
    df = df.join(zip_coords.add_prefix("customer_"), on="customer_zip_code_prefix")
    df = df.join(zip_coords.add_prefix("seller_"), on="seller_zip_code_prefix")

    # --- Feature engineering ---------------------------------------------------------------
    df["distance_km"] = haversine_km(
        df["seller_geolocation_lat"], df["seller_geolocation_lng"],
        df["customer_geolocation_lat"], df["customer_geolocation_lng"],
    )
    df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    # Freight is charged per seller shipment and split across its items, so the cart matters.
    df["items_in_order"] = df.groupby("order_id")["order_item_id"].transform("size")
    df["items_from_seller"] = df.groupby(["order_id", "seller_id"])["order_item_id"].transform("size")
    df["purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_month"] = df["purchase_timestamp"].dt.month
    df["product_category"] = (
        df["product_category_name_english"].fillna(df["product_category_name"]).fillna("unknown")
    )
    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    # Store every numeric feature as float: integer columns cannot hold missing values, which
    # would break the model's input schema when it is called later from an endpoint.
    df[NUMERIC] = df[NUMERIC].astype("float64")

    return df[FEATURES + [TARGET, "purchase_timestamp"]]


# -----------------------------------------------------------------------------------------
# 2. Models
# -----------------------------------------------------------------------------------------
def log_target(regressor):
    """Train on log(1 + freight) because freight is very skewed; predictions come back in R$."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def make_linear_model(alpha):
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
    ])
    return log_target(Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha))]))


def make_boosting_model(args):
    # Categorical columns come first, encoded as integers; gradient boosting splits on them natively.
    # Unseen categories become -1, which the model treats as missing.
    preprocess = ColumnTransformer([
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ])
    model = HistGradientBoostingRegressor(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return log_target(Pipeline([("preprocess", preprocess), ("model", model)]))


def evaluate(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_5_brl": float((errors <= 5).mean()),  # share of predictions within R$5 of the truth
    }


# -----------------------------------------------------------------------------------------
# 3. Plots
# -----------------------------------------------------------------------------------------
def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = float(np.percentile(y_true, 99))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, label="perfect prediction")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual freight (R$)", ylabel="Predicted freight (R$)",
           title="Test set: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_importance(model, X, y):
    sample = X.sample(min(5000, len(X)), random_state=42)
    result = permutation_importance(model, sample, y.loc[sample.index], scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=42, n_jobs=-1)
    order = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURES)[order], result.importances_mean[order])
    ax.set(xlabel="Increase in MAE (R$) when the feature is shuffled", title="Permutation feature importance")
    fig.tight_layout()
    return fig, dict(zip(FEATURES, result.importances_mean.round(4).tolist()))


# -----------------------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the Olist freight-cost model.")
    p.add_argument("--data", required=True, help="Folder containing the 9 Olist CSV files")
    p.add_argument("--test-start", default="2018-06-01", help="Orders from this date on form the test set")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--max-leaf-nodes", type=int, default=63)
    p.add_argument("--min-samples-leaf", type=int, default=40)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--output-dir", default="outputs", help="Folder for metrics.json (Azure ML keeps ./outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("data", "output_dir")})

        # ---- Data and time-based split ------------------------------------------------------
        df = build_dataset(args.data)
        cutoff = pd.Timestamp(args.test_start)
        train, test = df[df["purchase_timestamp"] < cutoff], df[df["purchase_timestamp"] >= cutoff]
        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]
        print(f"Rows: train={len(train):,} (before {args.test_start}), test={len(test):,}")
        mlflow.log_metrics({"rows_train": len(train), "rows_test": len(test)})

        # ---- Train and compare three models -------------------------------------------------
        candidates = {
            "baseline": DummyRegressor(strategy="median"),
            "linear": make_linear_model(args.ridge_alpha),
            "boosting": make_boosting_model(args),
        }
        results = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            results[name] = evaluate(y_test.to_numpy(), model.predict(X_test))
            mlflow.log_metrics({f"{name}_{metric}": value for metric, value in results[name].items()})
            print(f"{name:<9} MAE R${results[name]['mae']:.2f}  RMSE R${results[name]['rmse']:.2f}  "
                  f"R2 {results[name]['r2']:.3f}  within R$5: {results[name]['within_5_brl']:.1%}")

        best = candidates["boosting"]
        improvement = 1 - results["boosting"]["mae"] / results["baseline"]["mae"]
        mlflow.log_metric("mae_improvement_vs_baseline", improvement)
        print(f"Gradient boosting reduces MAE by {improvement:.1%} compared with the baseline.")

        # ---- Explain the model --------------------------------------------------------------
        predictions = best.predict(X_test)
        mlflow.log_figure(plot_predictions(y_test.to_numpy(), predictions), "plots/predicted_vs_actual.png")
        importance_fig, importances = plot_importance(best, X_test, y_test)
        mlflow.log_figure(importance_fig, "plots/feature_importance.png")
        mlflow.log_dict(importances, "feature_importance.json")

        # ---- Save the model (MLflow format -> ready to register and deploy) -----------------
        # save_model writes a plain MLflow model folder into ./outputs, which Azure ML uploads
        # with the job. (mlflow.sklearn.log_model in MLflow 3 calls a "logged models" API that
        # the Azure ML tracking server does not offer, so we avoid it.)
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({"test_start": args.test_start, "results": results, "feature_importance": importances}, f, indent=2)

        model_dir = os.path.join(args.output_dir, "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)  # save_model refuses to overwrite an existing folder
        signature = infer_signature(X_test.head(100), predictions[:100])
        mlflow.sklearn.save_model(
            best,
            model_dir,
            signature=signature,
            input_example=X_test.head(3),
            # Newer MLflow defaults to the "skops" format, which needs an extra package that the
            # curated environment does not include. cloudpickle works everywhere.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        print(f"Done. MLflow model saved to {model_dir}")


if __name__ == "__main__":
    main()
