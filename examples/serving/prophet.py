from serveml.api import ApiBuilder
from serveml.inputs import BasicInput
from serveml.loader import load_mlflow_model
from serveml.predictions import GenericPrediction

# load model
model = load_mlflow_model(
    # MlFlow model path
    "models:/prophet_model/1",
    # MlFlow Tracking URI
    "http://localhost:5000",
)


# Implement deserializer for input data
class PeriodPrediction(BasicInput):
    periods: int


# implement application
app = ApiBuilder(GenericPrediction(model), PeriodPrediction).build_api()
