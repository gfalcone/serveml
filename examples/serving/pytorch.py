from mlserve.api import ApiBuilder
from mlserve.inputs import BasicInput
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction

# load model
model = load_mlflow_model(
    # MlFlow model path
    "models:/pytorch_model/1",
    # MlFlow Tracking URI
    "http://localhost:5000",
)


# Implement deserializer for input data
class LinearRegression(BasicInput):
    input_prediction: float


# implement application
app = ApiBuilder(GenericPrediction(model), LinearRegression).build_api()
