## Proxy

When you deploy an API, there is usually a proxy involved between your application and the client (one of the most famous technology is NGINX)

Due to some [limitations on FastAPI](https://fastapi.tiangolo.com/advanced/sub-applications-proxy/), it is impossible to do path rewriting. 

Here is an example : 

Let's say you have a `serveml` application running and listening on **/**

Between this application and the client you normally define the proxy route to access to this application.    

In this case, we would say that we want to access this application when the route is **/api/sklearn_model/v1**

The thing is, these routes are not defined in our application, this is where the proxy comes in (in theory) and replaces the routes like this : 

- **/api/sklearn_model/v1/predict** -> **/predict**
- **/api/sklearn_model/v1/feedback** -> **/feedback**

If you want to do that kind of thing, you will have to define these routes on both the **proxy** and the **serveml application** 

Here is how to do it on `serveml` : 

````python hl_lines="35"
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
app = ApiBuilder(
	GenericPrediction(model), 
	WineComposition, 
	api_prefix="/api/sklearn_model/v1"
).build_api()
````