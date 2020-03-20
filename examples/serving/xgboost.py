from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction
from mlserve.api import ApiBuilder
from pydantic import BaseModel

# load model
model = load_mlflow_model(
    # MlFlow model path
    'models:/xgboost_model/1',
    # MlFlow Tracking URI
    'http://localhost:5000',
)


# Implement deserializer for input data
class PetalComposition(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# implement application
app = ApiBuilder(GenericPrediction(model), PetalComposition).build_api()
