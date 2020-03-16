from mlserve.api import ApiBuilder
from mlserve.loader import load_mlflow_model
from mlserve.ml.sklearn import SklearnModel
from pydantic import BaseModel

# load model
model = load_mlflow_model(
    # MlFlow model path
    'models:/my_model/Production',
    # MlFlow Tracking URI (optional)
    'http://localhost:5000',
)


# Implement deserializer for input data
class WineComposition(BaseModel):
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
app = ApiBuilder(SklearnModel(model), WineComposition).build_api()
