"""Compare a median baseline with one freight regressor in an Azure ML command job."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CATEGORICAL = ["seller_state", "customer_state", "product_category"]
NUMERIC = [
    "price", "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "volume_cm3", "distance_km", "same_state",
    "items_in_order", "items_from_seller", "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC


def read_table(folder: Path) -> pd.DataFrame:
    # Azure ML mounts each MLTable asset folder; local tests use the same layout.
    return pd.read_csv(folder / "orders.csv")


def make_model(name: str):
    if name == "ridge":
        prep = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numbers", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
        ])
        regressor = Ridge(alpha=10)
    else:
        prep = ColumnTransformer([
            ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("numbers", "passthrough", NUMERIC),
        ])
        regressor = HistGradientBoostingRegressor(
            categorical_features=list(range(len(CATEGORICAL))),
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
            random_state=42,
        )
    pipeline = Pipeline([("prepare", prep), ("regressor", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def metrics(y, predictions):
    return {
        "mae_brl": float(mean_absolute_error(y, predictions)),
        "rmse_brl": float(np.sqrt(mean_squared_error(y, predictions))),
        "within_5_brl": float((np.abs(y - predictions) <= 5).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "boosting"], default="ridge")
    args = parser.parse_args()
    train = read_table(args.train)
    valid = read_table(args.validation)
    for part in (train, valid):
        assert part.freight_value.notna().all() and part.freight_value.ge(0).all()
        assert set(FEATURES).issubset(part.columns)

    x_train, y_train = train[FEATURES], train.freight_value.to_numpy()
    x_valid, y_valid = valid[FEATURES], valid.freight_value.to_numpy()
    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    model = make_model(args.model)

    with mlflow.start_run():
        mlflow.log_params({"algorithm": args.model, "train_rows": len(train), "validation_rows": len(valid)})
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        baseline_metrics = metrics(y_valid, baseline.predict(x_valid))
        model_metrics = metrics(y_valid, predictions)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metrics({f"validation_{k}": v for k, v in model_metrics.items()})
        print("Training median baseline:", baseline_metrics)
        print(f"{args.model} validation:", model_metrics)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_valid[:3000], predictions[:3000], s=4, alpha=0.2)
        ax.plot([0, 100], [0, 100], color="black", linewidth=1)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Actual freight (R$)",
               ylabel="Predicted freight (R$)", title=f"{args.model}: validation items")
        mlflow.log_figure(fig, "plots/validation_predictions.png")
        plt.close(fig)

        model_dir = Path("outputs") / f"{args.model}-model"
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(
            model, str(model_dir),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        print(f"Model saved to {model_dir}. Test data remains untouched.")


if __name__ == "__main__":
    main()
