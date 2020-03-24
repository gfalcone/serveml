import os

import mlflow

from mlserve.api import ApiBuilder
from mlserve.inputs import BasicInput
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction

# getting run_id
mlflow_client = mlflow.tracking.MlflowClient("http://localhost:5000")
run_id = mlflow_client.list_run_infos(experiment_id=4)[0].run_id
current_directory = os.getcwd()

model = load_mlflow_model(
    "{}/4/{}/artifacts/model".format(current_directory, run_id)
)


# Implement deserializer for input data
class PetalComposition(BasicInput):
    SepalWidth: float
    SepalLength: float
    PetalLength: float
    PetalWidth: float


# implement application
app = ApiBuilder(GenericPrediction(model), PetalComposition).build_api()
