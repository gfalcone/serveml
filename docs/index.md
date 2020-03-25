# serveml

`serveml` is a Python library that helps you package your Machine Learning model easily into a REST API.

The idea behind `serveml` is to define a set of generic endpoints to make predictions easily !

## Philosophy

The recent surge of interest in harnessing Machine Learning to solve business problems has shed the light upon how hard it is to put it into production.

Data Scientists often struggle with the technology gap between their development environment and the production environment for a lot of reasons (scalability, notebook code not ready for production, ...). 

The goal of `serveml` is to reduce this gap by offering a simple way to package the model behind a REST API.

`serveml` is not the first library to help packaging of ML models behind API, but aims to be a complete one. 

The main goals of this library is to :

* Have an API that could be used by everybody (with documentation)
* Make the API more reliable by verifying the request's input
* Have an easy way to retrieve Machine Learning projects


These points are mainly solved thanks to : 

* [FastAPI](https://fastapi.tiangolo.com/), as it offers a clean and eay way to define a documented API.
* [MLflow](https://mlflow.org/docs/latest/index.html), which helps Data Scientists store their training models and artifacts (data for example) in a generic way.

## Libraries supported

`serveml` is designed to support any Machine Learning library you want to use.

As it is tightly coupled with MLflow to handle model retrieving, it supports all libraries supported by MLflow out-of-the-box.'

If you want to find more on libraries supported by MLflow, go here : [Libraries supported by MLflow](https://mlflow.org/docs/latest/models.html#id10)

If the library you wish to use is not supported by MLflow, do not worry, this case is covered [here](https://www.mlflow.org/docs/latest/models.html#model-customization)  

## Requirements

Python 3.6+ (for FastAPI)

## Installation

```bash
pip install serveml
```