import configparser

from fastapi import FastAPI

from mlserve.ml import AbstractModel


class ApiBuilder(object):
    """
    Main class for generating an API thanks to FastAPI and Pydantic.
    """
    def __init__(
            self,
            model: AbstractModel,
            input_class,
            configuration_path=None,
    ):
        """
        :param model: <mlserve.ml.model.AbstractModel> object that inplements
        helper functions to have a proper API working.
        :param input_class: <pydantic.BaseModel> object that implements the
        `/predict` input validator
        """
        self.model = model
        self.input_class = input_class
        self.configuration = configparser.ConfigParser()
        self.load_configuration(configuration_path)

    def load_configuration(self, configuration_path):
        if configuration_path is not None:
            self.configuration.read(configuration_path)

    def build_api(self, kwargs: dict = None):
        """
        This function actually defines endpoint for our API, namely the
        `/predict` endpoint and the `/feedback` endpoint.
        """
        # retrieve fastapi configuration
        print(self.configuration)
        fastapi_configuration = (
            self.configuration['fastapi']
            if 'fastapi' in self.configuration.sections() else {}
        )

        # to override parameters in configuration file
        if kwargs is not None and 'fastapi' in kwargs.keys():
            fastapi_configuration.update(kwargs.get('fastapi'))

        app = FastAPI(**fastapi_configuration)

        # adding a route for predict
        input_class = self.input_class

        @app.post("/predict/")
        async def predict(input: input_class):
            return self.model.predict(input)

        return app
