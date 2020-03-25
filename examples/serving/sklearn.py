from serveml.api import ApiBuilder
from serveml.inputs import BasicInput
from serveml.loader import load_mlflow_model
from serveml.predictions import GenericPrediction


# load model
model = load_mlflow_model(
    # MlFlow model path
    "models:/sklearn_model/1",
    # MlFlow Tracking URI
    "http://localhost:5000",
)


# Implement deserializer for input data
class WineComposition(BasicInput):
    alcohol: float
    chlorides: float
    citric_acid: float
    density: float
    fixed_acidity: float
    free_sulfur_dioxide: int
    pH: float
    residual_sugar: float
    sulphates: float
    total_sulfur_dioxide: int
    volatile_acidity: int


# implement application
app = ApiBuilder(GenericPrediction(model), WineComposition).build_api()
