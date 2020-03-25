from serveml.api import ApiBuilder
from serveml.inputs import BasicInput
from serveml.loader import load_mlflow_model
from serveml.predictions import GenericPrediction

# load model
model = load_mlflow_model(
    # MlFlow model path
    "models:/xgboost_model/1",
    # MlFlow Tracking URI
    "http://localhost:5000",
)


# Implement deserializer for input data
class PetalComposition(BasicInput):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# implement application
app = ApiBuilder(GenericPrediction(model), PetalComposition).build_api()
