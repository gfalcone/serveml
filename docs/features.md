# Features

## Machine Learning

`mlserve` is designed to support any Machine Learning library, and is already ready to be customized as you wish. 

One of the main thing Data Scientists struggle with is that they do not version their models.

[MLflow](https://mlflow.org/docs/latest/index.html) already defines a way to store and retrieve models you trained and offers an easy way to version your models

In order to expose any kind of Machine Learning model we defined two functions that might help you going faster in your deployment !

### Getting a model

In order to get a model, we rely heavily on MLflow, and the simplest way to do it is this way : 

```Python
from mlserve.loader import load_mlflow_model


model = load_mlflow_model(
    # MlFlow model path
    'models:/sklearn_model/1',
    # MlFlow Tracking URI
    'http://localhost:5000',
)
```

This will retrieve the first version of the **sklearn_model** in [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)

Note that you can use the following syntaxes for loading the model : [MLflow Model Loading Syntaxes](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model)

### Handling the prediction

Handling the API - Machine Learning Model interface can be tricky. APIs often exchange data in JSON format, whereas the main format for Machine Learning predictions is a **pandas.DataFrame**

In order to avoid boilerplate code, we defined a **GenericPrediction** object that handles this interface for you. 

Here is an example on how to use it :

```Python
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction


# load model
model = load_mlflow_model(
    # MlFlow model path
    'models:/sklearn_model/1',
    # MlFlow Tracking URI
    'http://localhost:5000',
)

# implement prediction methods
generic_prediction = GenericPrediction(model)
```

This **GenericPrediction** object will define : 

* Method to transform **JSON** into **pandas.DataFrame**
* Method to apply prediction
* Method to transform prediction result into **JSON**

## API

### Defining the API

Defining the same endpoints for every Machine Learning model you'll put into production would be painful to do every time.   

We defined two endpoints for you in order to avoid doing that : 

* **/predict** to do a prediction based on the pre-trained model 
* **/feedback** in order to get a feedback of the prediction done (*EXPERIMENTAL*)

The only thing you need to do is define the type of data you will provide to the **/predict** endpoint : 

```Python
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
app = ApiBuilder(GenericPrediction(model), WineComposition).build_api()
```

Now you have a FastAPI application, you can run it thanks to [Uvicorn](https://www.uvicorn.org/).

Let's say you saved the preceding script into a file named `api.py`, here is a simple command to run it : 

```bash
uvicorn api:app --host 0.0.0.0
```

### Using the API

[FastAPI](https://fastapi.tiangolo.com/) defines the documentation for you and provides two interfaces for that 

#### ReDoc

You can access it on [http://localhost:8000/redoc]()

![alt text](./images/redoc.gif "ReDoc interface")


#### OpenAPI

You can access it on [http://localhost:8000/docs]()

![alt text](./images/docs.gif "OpenAPI interface")
