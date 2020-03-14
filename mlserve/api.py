from fastapi import FastAPI

from mlserve.ml.model import AbstractModel


class ApiBuilder(object):
    def __init__(
            self,
            model: AbstractModel,
            input_class,
            fast_api_options: dict,
    ):
        self.model = model
        self.input_class = input_class
        self.fast_api_options = fast_api_options

    def build_api(self):
        app = FastAPI(**self.fast_api_options)

        # adding a route for predict
        Input = self.input_class

        @app.post("/predict/")
        async def predict(input: Input):
            return self.model.predict(input)

        return app
