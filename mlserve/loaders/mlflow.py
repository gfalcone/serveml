from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model


class MlflowModelLoader(object):
    """
    Class for loading MLflow models
    """

    def __init__(self, tracking_uri):
        super().__init__()
        self.mlflow_client = MlflowClient(tracking_uri)

    def load_model(self, path):
        return load_model(path)
