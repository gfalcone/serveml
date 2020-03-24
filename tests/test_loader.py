import os
import unittest

import mlflow

from sklearn.linear_model import ElasticNet

from mlflow.exceptions import RestException
from mlflow.tensorflow import _TF2Wrapper

from mlserve.loader import load_mlflow_model


class TestMLflowModelLoader(unittest.TestCase):
    def test_load_model_from_mlflow_server(self):
        model = load_mlflow_model(
            "models:/sklearn_model/1", "http://localhost:5000"
        )
        self.assertIsInstance(model, ElasticNet)

    def test_load_model_from_run_id(self):
        # getting run_id
        mlflow_client = mlflow.tracking.MlflowClient("http://localhost:5000")
        run_id = mlflow_client.list_run_infos(experiment_id=4)[0].run_id
        current_directory = os.getcwd()
        model = load_mlflow_model(
            "{}/4/{}/artifacts/model".format(current_directory, run_id)
        )
        self.assertIsInstance(model, _TF2Wrapper)

    def test_load_unexisting_model(self):
        with self.assertRaises(RestException):
            load_mlflow_model(
                "models:/unexisting_model/1", "http://localhost:5000"
            )
