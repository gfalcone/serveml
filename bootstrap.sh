#!/usr/bin/env bash
set -e

mlflow server \
  --backend-store-uri sqlite:///database.db \
  --default-artifact-root file:///app/ \
  --host 0.0.0.0 &
sleep 2

case "$1" in
  test)
    pytest
    ;;
  *)
    exec "$@"
    ;;
esac
