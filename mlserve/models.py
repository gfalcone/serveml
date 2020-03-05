import mlflow
from mlflow.tracking import MlflowClient


class Model(object):
    """
    Base class for loading a model
    """

    def __init__(self):
        pass

    def load_model(self, path):
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

    def __init__(self, tracking_uri):
        super().__init__()
        self.mlflow_client = MlflowClient(tracking_uri)

    def load_model(self, model_name):
        items = self.mlflow_client.list_registered_models()
        registered_model = filter(lambda x: x.name == model_name, items)

        if len(registered_model) == 1:
            return registered_model[0]

        raise ValueError('Could not find ')


if __name__ == '__main__':
    model = MlflowModel('http://localhost:5000')
    print(model.mlflow_client.list_registered_models())
    print(model.load_model('my_model'))
