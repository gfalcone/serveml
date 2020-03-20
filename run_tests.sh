python -m unittest tests.test_api
python -m unittest tests.test_data_models
python -m unittest tests.test_io
python -m unittest tests.test_loader
python -m unittest tests.test_predictions

# serving for each library (except tensorflow based)
python -m unittest tests.examples.serving.test_prophet
python -m unittest tests.examples.serving.test_pytorch
python -m unittest tests.examples.serving.test_sklearn
python -m unittest tests.examples.serving.test_xgboost