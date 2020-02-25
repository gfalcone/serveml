from mlflow.tracking import MlflowClient


class Model(object):
    """
    Base class for loading a model
    """

    def __init__(self):
        pass

    def load_model(self):
        """
        Base function for loading model
        :return:
        """
        pass

    def predict(self, data):
        """
        Base function for prediction
        :param data:
        :return:
        """
        pass


class LocalModel(object):
    """
    Class for loading custom models
    """

    def __init__(self, path):
        super().__init__()
        self.path = self.path
        self.model = self.load_model(path)

    def load_model(self, path):
        pass

    def predict(self, data):
        pass


class MlflowModel(Model):
    """
    Class for loading MLflow models
    """

    def __init__(self, name):
        super().__init__(self, name)
        self.mlflow_client = MlflowClient()

    def load_model(self, run_id):
        self.mlflow_client.get_model_version_download_uri()
