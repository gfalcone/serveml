# mlserve

`mlserve` is a Machine Learning tool that helps you package your model easily into an API and deploy it easily into production ! 

Doing that is no easy task, and is not really the Data Scientist's role.

With `mlserve` you can abstract all of technical challenges around building an API and ship your model easily !


## Requirements

Docker
mlflow

## How to package ? 

First of all, you need to define your API into a Python file:

```python
from mlserve.api import ApiBuilder
from mlserve.ml.model import AbstractModel
from mlserve.io import pydantic_model_to_pandas, pandas_to_dict
from mlserve.mlflow import MlflowModelLoader
from pydantic import BaseModel

# Set MLFlow model loader and load model
model_loader = MlflowModelLoader('http://localhost:5000')
model = model_loader.load_model('models:/my_model/Production')


# Implement methods for prediction
class SklearnModel(AbstractModel):
    def _transform_input(self, input):
        return pydantic_model_to_pandas(input)

    def _apply_model(self, transformed_input):
        return self.model.predict(transformed_input)

    def _transform_output(self, output):
        # we get a ndarray in output, converting it to a dictionary
        return {'quality': output.item(0)}


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


# Build API
fast_api_options = {
    'title': 'WineQualityApi',
    'description': 'This API helps you determine if the quality of the Wine is good or not',  # NOQA
    'version': '0.1.0'
}

app = ApiBuilder(
    SklearnModel(model),
    WineComposition,
    fast_api_options
).build_api()
```

Then package it with this command (this will trigger the docker build):

```bash
mlserve package iris_api
```

Now that your api is built, you can test it in local ! 

```bash
mlserve test start iris_api
```

This should normally gives you back the IP and the port of your API so you can request it ! 

When you're done with it you can stop it (by default, it will be active for 10 minutes)

```bash
mlserve test stop iris_api
```

When you're ready to go, you can deploy it like this to your repository manager : 

```bash
mslerve store iris_api
```

And then deploy it to your favorite cloud provider : 

```bash
mlserve deploy iris_api
```
