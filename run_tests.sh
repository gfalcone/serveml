#!/usr/bin/env bash

export BUILD_DIRECTORY=$(pwd)
mlflow server \
  --backend-store-uri sqlite:///database.db \
  --default-artifact-root file://$BUILD_DIRECTORY \
  --host 0.0.0.0 &
sleep 2
export MLFLOW_TRACKING_URI=http://localhost:5000
pytest