"""Train a late-delivery classifier and save a sklearn-flavored MLflow model."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["customer_state", "seller_state"]
NUMERIC = [
    "promised_days", "distance_km", "n_items", "total_price",
    "total_weight_g", "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "late_delivery"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()
    train = pd.read_parquet(args.train / "data.parquet")
    test = pd.read_parquet(args.test / "data.parquet")
    assert train[FEATURES + [TARGET]].notna().all().all()
    assert test[FEATURES + [TARGET]].notna().all().all()

    prep = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("number", StandardScaler(), NUMERIC),
    ])
    model = Pipeline([
        ("prepare", prep),
        ("classify", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])
    model.fit(train[FEATURES], train[TARGET])
    y = test[TARGET]
    predicted = model.predict(test[FEATURES])
    probability = model.predict_proba(test[FEATURES])[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    scores = {
        "test_roc_auc": float(roc_auc_score(y, probability)),
        "test_average_precision": float(average_precision_score(y, probability)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "test_late_precision": float(precision),
        "test_late_recall": float(recall),
        "test_late_f1": float(f1),
        "test_false_negatives": int(fn),
        "test_false_positives": int(fp),
        "test_true_positives": int(tp),
        "test_true_negatives": int(tn),
        "test_orders": len(test),
    }
    with mlflow.start_run():
        mlflow.log_param("positive_class", "late_delivery=1")
        mlflow.log_param("train_orders", len(train))
        mlflow.log_metrics(scores)
        model_dir = Path("outputs/model")
        model_dir.parent.mkdir(exist_ok=True)
        mlflow.sklearn.save_model(
            model, str(model_dir),
            signature=infer_signature(test[FEATURES].head(20), predicted[:20]),
            input_example=test[FEATURES].head(2),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    print("Future-period late rate:", round(y.mean(), 4))
    print("All future-period scores:", scores)
    print("Saved sklearn-flavored MLflow model to", model_dir)


if __name__ == "__main__":
    main()
