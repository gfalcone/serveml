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
from mlserve.models import MlflowModel
from mlserve.api import Api

model = MlflowModel('iris_classifier')

api = Api('iris_api', model, validator)

api.serve()
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
