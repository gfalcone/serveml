# mlserve

`mlserve` is a Machine Learning tool that helps you exposer your Machine Learning model easily into an API.

Doing that is no easy task, and is not really the Data Scientist's role.

With `mlserve` you can abstract all of technical challenges around building an API and ship your model easily !

## Prerequisites 

Have a MlFlow server running, in order to have one on your local setup : 

```bash
mlflow server --backend-store-uri mysql://root:root@mysql/mlflow --default-artifact-root s3://drivy-data-dev/mlflow/app/runs -h 0.0.0.0
```

## How to use ? 

First of all, you need to define your API into a Python file, say `api.py`

```python
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
```

Then to run it : 

```bash
uvicorn api:app --host 0.0.0.0
```

You can now access your API's documentation, generated by [redoc](https://github.com/Redocly/redoc) on [localhost:8000/redoc]() :

![Redoc Interface](https://github.com/gfalcone/mlserve/docs/images/redoc.png)

Or again access your API with Swagger on [localhost:8000/docs]() :

![Swagger Interface](https://github.com/gfalcone/mlserve/docs/images/swagger.png)
