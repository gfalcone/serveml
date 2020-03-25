#!/usr/bin/env bash

export BUILD_DIRECTORY=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///database.db \
  --default-artifact-root file://$BUILD_DIRECTORY \
  --host 0.0.0.0 &
sleep 2
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m examples.training.sklearn
python -m examples.training.pytorch
python -m examples.training.keras
python -m examples.training.xgboost
python -m examples.training.tensorflow
python -m examples.training.prophet
