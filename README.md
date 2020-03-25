# mlserve

[![Build Status](https://travis-ci.org/gfalcone/mlserve.svg?branch=master)](https://travis-ci.org/gfalcone/mlserve)

`mlserve` is a Python library that helps you package your Machine Learning model easily into a REST API.

The idea behind `mlserve` is to define a set of generic endpoints to make predictions easily !

## Requirements

- Python 3.6+
- [FastAPI](https://fastapi.tiangolo.com/) (for the API part)
- [MLflow](https://mlflow.org/) (for model loading)
- [Uvicorn](https://www.uvicorn.org/) (to run api)


## Installation

```bash
pip install mlserve
```

## Documentation

You can find the full documentation here : https://gfalcone.github.io/mlserve/

## How to use ? 

### Prerequisites 

In order to run the examples we put, you'll need an MLflow server running. 

As we do not expect you to have already this in place, we set up a docker container in order to speed things up.

You'll need to do the following things to set up MLflow on your local machine : 

```bash
git clone https://github.com/gfalcone/mlserve
cd mlserve
docker-compose build
docker-compose up
```

### Training

First of all, you need to have a model already trained and registered in MlFlow

Luckily for you, we already have a set of examples that you can already use.

Let's say you have a scikit-learn model, like this one (taken from examples/serving/sklearn.py): 


```python
"""
Example taken from https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py
"""
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    experiment_name = 'test_sklearn'
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=1):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="sklearn_model"
        )
```

You can run it with : 

```bash
python -m examples.training.sklearn
```

### Serving

We can then define the API this way (taken from examples/serving/sklearn.py): 

```python
from mlserve.api import ApiBuilder
from mlserve.inputs import BasicInput
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction

# load model
model = load_mlflow_model(
    # MlFlow model path
    'models:/sklearn_model/1',
    # MlFlow Tracking URI (optional)
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

And then run it with : 

```bash
uvicorn examples.serving.sklearn:app --host 0.0.0.0
```

You can now access your API's documentation, generated by [redoc](https://github.com/Redocly/redoc) on [localhost:8000/redoc]() or  access your API with Swagger on [localhost:8000/docs]() :

![API](https://github.com/gfalcone/mlserve/blob/master/docs/images/mlserve-example.gif)

Don't forget to exit the Docker container to shut down MLflow when you're done (with Ctrl+C)

## Testing

### Unit tests

To run unit tests, do the following : 

```bash
docker build --tag=mlserve -f Dockerfile .
```

### Documentation

If you want to look how the documentation will be rendered after making changes to it : 

```bash
pip install -r requirements-doc.txt
mkdocs serve
```

## Contributing

If you wish to make some changes, we are obviously open to Pull Requests. 

Please not that in order for your PR to be merged the following points are mandatory : 

- The code must be formatted with [Black](https://github.com/psf/black), here is the command to use to reformat your code : 
```bash
black . -l 79
```
- CI must be green on Travis