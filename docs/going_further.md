## FastAPI configuration

By default, we do not change the FastAPI configuration, so you end up with a generic configuration.

One way to avoid that is to define a configuration file (for example `fastapi.cfg`) like this : 

```
[fastapi]
title = WineQualityApi
description = This API helps you determine if the quality of the Wine is good or not
version = 0.1.0
```

And then to give it to your application : 

```python
from mlserve.api import ApiBuilder
from mlserve.inputs import BasicInput
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction


# load model
model = load_mlflow_model(
    # MlFlow model path
    'models:/sklearn_model/1',
    # MlFlow Tracking URI
    'http://localhost:5000',
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
app = ApiBuilder(GenericPrediction(model), WineComposition, configuration_path='fastapi.cfg').build_api()
```

## Defining your own prediction object

The **GenericPrediction** object implements the **AbstractPrediction** class in order to ease the interface between the API and the model prediction.

Under the hood, the **GenericPrediction** does a little a bit more than that, in order : 

- *_transforms_input*: Transforms **JSON** into **pandas.DataFrame** 
- *_fetch_data*: Fetch data from an external source (by default does nothing)
- *_combine_fetched_data_with_input*: Combine fetched data with **pandas.DataFrame** (by default only returns **pandas.DataFrame**)
- *_apply_model*: Apply model 
- *_transform_output*: Transforms result (either **pandas.DataFrame**, **numpy.ndarray**, **pandas.Series**) into JSON

### Retrieving additional data before applying model

Let's say that the data given in input of the API is not enough, and you need additional information in order to make your prediction. 

This could be done easily by overriding these two functions in the **GenericPrediction** class (inherited by the **AbstractPrediction** class): 


```python hl_lines="27 28 29 30 31 32 33 34 35 36 37"
{!./mlserve/predictions.py!}
```
