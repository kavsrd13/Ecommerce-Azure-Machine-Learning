"""One Azure ML sweep trial: train logistic regression, log validation metrics."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["customer_state", "seller_state"]
NUMERIC = [
    "promised_days", "distance_km", "n_items", "total_price",
    "total_weight_g", "purchase_month",
]
FEATURES = CATEGORICAL + NUMERIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--positive-weight", type=float, required=True)
    parser.add_argument("--model-output", type=Path, default=Path("outputs/model"))
    args = parser.parse_args()

    train = pd.read_parquet(args.train / "data.parquet")
    validation = pd.read_parquet(args.validation / "data.parquet")
    assert train[FEATURES + ["late_delivery"]].notna().all().all()
    assert validation[FEATURES + ["late_delivery"]].notna().all().all()
    model = Pipeline([
        ("prepare", ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("number", StandardScaler(), NUMERIC),
        ])),
        ("classify", LogisticRegression(
            C=args.c, class_weight={0: 1, 1: args.positive_weight},
            max_iter=500, random_state=42,
        )),
    ])
    model.fit(train[FEATURES], train.late_delivery)
    probability = model.predict_proba(validation[FEATURES])[:, 1]
    predicted = model.predict(validation[FEATURES])
    ap = average_precision_score(validation.late_delivery, probability)
    precision = precision_score(validation.late_delivery, predicted, zero_division=0)
    recall = recall_score(validation.late_delivery, predicted, zero_division=0)
    with mlflow.start_run():
        mlflow.log_params({"C": args.c, "positive_weight": args.positive_weight})
        mlflow.log_metric("validation_average_precision", ap)
        mlflow.log_metric("validation_late_precision", precision)
        mlflow.log_metric("validation_late_recall", recall)
        model_dir = args.model_output
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(
            model, str(model_dir),
            signature=infer_signature(validation[FEATURES].head(20), predicted[:20]),
            input_example=validation[FEATURES].head(2),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    print(f"C={args.c} positive_weight={args.positive_weight} validation AP={ap:.4f} precision={precision:.4f} recall={recall:.4f}")


if __name__ == "__main__":
    main()
