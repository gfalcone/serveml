import mlflow

from mlflow.pyfunc import load_model


def load_mlflow_model(path: str, tracking_uri: str = None):
    """
    Generic function that loads model from MlFlow.
    Relies on this function : https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model

    :param tracking_uri: MlFlow Tracking URI (example: http://localhost:5000)
    :param path: path of the model, can be one of the following:
        - ``/Users/me/path/to/local/model``
        - ``relative/path/to/local/model``
        - ``s3://my_bucket/path/to/model``
        - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
        - ``models:/<model_name>/<model_version>``
        - ``models:/<model_name>/<stage>``
    """
    if mlflow.set_tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    return load_model(path)
