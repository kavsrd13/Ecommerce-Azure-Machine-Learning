"""
STEP 2 of the pipeline - TRAINING.

Predicts DELIVERY TIME (days from purchase to delivery) of an Olist order from information
known at checkout. Reads orders_features.csv produced by the prep step.

    python train.py --features "<folder with orders_features.csv>" --model_output "<folder>"

Compared on the same time-based test set (orders from --test-start, default 2018-06-01):
  baseline       - always predicts the median delivery time of the training data
  olist_promise  - the delivery date Olist itself promised (a real business benchmark)
  linear         - ridge regression on log(delivery days)
  boosting       - gradient boosting with absolute-error loss = the EXPECTED delivery time (kept)
  boosting_p90   - gradient boosting with quantile loss = a SAFE PROMISE (90% arrive earlier)

Everything is logged with MLflow; the kept model is saved in MLflow format to --model_output.
"""

import argparse
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")  # no screen on a cloud VM
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "delivery_days"
CATEGORICAL = ["seller_state", "customer_state", "product_category"]
NUMERIC = [
    "distance_km", "same_state", "promised_days", "n_items", "n_sellers", "total_price", "total_freight",
    "total_weight_g", "total_volume_cm3", "purchase_month", "purchase_weekday", "purchase_hour",
]
FEATURES = CATEGORICAL + NUMERIC


# -----------------------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------------------
def boosting(loss, args, quantile=None):
    """Gradient boosting. loss='absolute_error' predicts the median, 'quantile' a percentile."""
    preprocess = ColumnTransformer([
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ])
    params = dict(
        loss=loss,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    if quantile is not None:
        params["quantile"] = quantile
    return Pipeline([("preprocess", preprocess), ("model", HistGradientBoostingRegressor(**params))])


def linear(alpha):
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
    ])
    pipe = Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha))])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def evaluate(y_true, y_pred):
    errors = y_pred - y_true
    return {
        "mae_days": float(mean_absolute_error(y_true, y_pred)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_3_days": float((np.abs(errors) <= 3).mean()),
        # Share of orders that arrive AFTER the predicted date: what a customer would call "late"
        "late_vs_prediction": float((y_true > y_pred).mean()),
    }


# -----------------------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------------------
def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = float(np.percentile(y_true, 99))
    ax.scatter(y_true, y_pred, s=4, alpha=0.2)
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, label="perfect prediction")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual delivery time (days)",
           ylabel="Predicted delivery time (days)", title="Test set: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_monthly(df, predictions, cutoff):
    """Actual vs predicted average delivery time per purchase month - shows data drift."""
    frame = pd.DataFrame({"month": df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp(),
                          "actual": df[TARGET], "predicted": predictions})
    monthly = frame.groupby("month").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly["actual"], marker="o", label="actual")
    ax.plot(monthly.index, monthly["predicted"], marker="o", label="predicted")
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1, label="train | test")
    ax.set(ylabel="Average delivery time (days)", title="Delivery time by purchase month")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_importance(model, X, y):
    sample = X.sample(min(5000, len(X)), random_state=42)
    result = permutation_importance(model, sample, y.loc[sample.index], scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=42, n_jobs=-1)
    order = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURES)[order], result.importances_mean[order])
    ax.set(xlabel="Increase in MAE (days) when the feature is shuffled", title="Permutation feature importance")
    fig.tight_layout()
    return fig, dict(zip(FEATURES, result.importances_mean.round(4).tolist()))


# -----------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the Olist delivery-time model.")
    p.add_argument("--features", required=True, help="Folder containing orders_features.csv (prep output)")
    p.add_argument("--model_output", default="outputs/model", help="Folder to save the MLflow model to")
    p.add_argument("--test-start", default="2018-06-01", help="Orders from this date on form the test set")
    p.add_argument("--test-end", default="2018-09-01", help="Orders from this date on are ignored (incomplete data)")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(args.features, "orders_features.csv"), parse_dates=["order_purchase_timestamp"])
    df[NUMERIC] = df[NUMERIC].astype("float64")
    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("features", "model_output")})

        # ---- Time-based split -----------------------------------------------------------------
        start, end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
        ts = df["order_purchase_timestamp"]
        train, test = df[ts < start], df[(ts >= start) & (ts < end)]
        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]
        print(f"Orders: train={len(train):,} (before {args.test_start}), test={len(test):,}")
        mlflow.log_metrics({"rows_train": len(train), "rows_test": len(test)})

        # ---- Benchmarks that need no training ---------------------------------------------------
        results = {
            "baseline": evaluate(y_test.to_numpy(), np.full(len(test), y_train.median())),
            "olist_promise": evaluate(y_test.to_numpy(), test["promised_days"].to_numpy()),
        }

        # ---- Train the models ---------------------------------------------------------------------
        models = {
            "linear": linear(args.ridge_alpha),
            "boosting": boosting("absolute_error", args),
            "boosting_p90": boosting("quantile", args, quantile=0.9),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            results[name] = evaluate(y_test.to_numpy(), model.predict(X_test))

        for name, metrics in results.items():
            mlflow.log_metrics({f"{name}_{metric}": value for metric, value in metrics.items()})
            print(f"{name:<14} MAE {metrics['mae_days']:5.2f} days  within 3 days {metrics['within_3_days']:6.1%}  "
                  f"late vs prediction {metrics['late_vs_prediction']:6.1%}")

        best = models["boosting"]
        improvement = 1 - results["boosting"]["mae_days"] / results["baseline"]["mae_days"]
        mlflow.log_metric("mae_improvement_vs_baseline", improvement)
        print(f"Gradient boosting reduces MAE by {improvement:.1%} compared with the baseline.")

        # ---- Explain ---------------------------------------------------------------------------------
        predictions = best.predict(X_test)
        mlflow.log_figure(plot_predictions(y_test.to_numpy(), predictions), "plots/predicted_vs_actual.png")
        recent = df[(ts >= "2017-06-01") & (ts < end)]
        mlflow.log_figure(plot_monthly(recent, best.predict(recent[FEATURES]), start), "plots/monthly_drift.png")
        importance_fig, importances = plot_importance(best, X_test, y_test)
        mlflow.log_figure(importance_fig, "plots/feature_importance.png")
        mlflow.log_dict({"results": results, "feature_importance": importances}, "metrics_summary.json")

        # ---- Save the kept model in MLflow format ----------------------------------------------------
        # Save to a temporary folder first, then copy into the output folder (Azure ML creates the
        # output folder in advance, and save_model refuses to write into an existing folder).
        signature = infer_signature(X_test.head(100), predictions[:100])
        with tempfile.TemporaryDirectory() as tmp:
            local_model = os.path.join(tmp, "model")
            mlflow.sklearn.save_model(
                best, local_model, signature=signature, input_example=X_test.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            os.makedirs(args.model_output, exist_ok=True)
            shutil.copytree(local_model, args.model_output, dirs_exist_ok=True)
        print(f"Done. MLflow model saved to {args.model_output}")


if __name__ == "__main__":
    main()
