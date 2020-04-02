"""
Example taken from https://github.com/mlflow/mlflow/blob/master/examples/xgboost/train.py
"""

import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb


import mlflow
import mlflow.xgboost


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    experiment_name = "test_xgboost"

    mlflow.set_tracking_uri("http://localhost:5000")

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=4):

        # train model
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

        mlflow.xgboost.log_model(
            model, "model", registered_model_name="xgboost_model"
        )


if __name__ == "__main__":
    main()
