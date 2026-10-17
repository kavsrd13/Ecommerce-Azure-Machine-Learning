"""Evaluate one selected Azure ML freight model on the held-out future period."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from train import FEATURES, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()
    test = pd.read_csv(args.test / "orders.csv")
    model = mlflow.sklearn.load_model(str(args.model_dir))
    predictions = model.predict(test[FEATURES])
    scores = metrics(test.freight_value.to_numpy(), predictions)
    with mlflow.start_run():
        mlflow.log_param("evaluation_window", "2018-06-01 to 2018-08-31")
        mlflow.log_metric("test_rows", len(test))
        mlflow.log_metrics({f"test_{name}": value for name, value in scores.items()})
    print(f"Held-out test: {len(test):,} items. Metrics:", scores)


if __name__ == "__main__":
    main()
