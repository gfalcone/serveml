import configparser

from uuid import uuid4, UUID

from fastapi import FastAPI

from mlserve.predictions import AbstractPrediction
from mlserve.inputs import FeedbackInput, BasicInput


class ApiBuilder(object):
    """
    Main class for generating an API thanks to FastAPI and Pydantic.
    """
    def __init__(
            self,
            model: AbstractPrediction,
            predict_input_class,
            feedback_input_class=FeedbackInput,
            configuration_path: str = None,
    ) -> None:
        """
        :param model: <mlserve.ml.model.AbstractModel> object that inplements
        helper functions to have a proper API working.
        :param input_class: <pydantic.BaseModel> object that implements the
        `/predict` input validator
        """
        self.model = model
        self.predict_input_class = predict_input_class
        self.feedback_input_class = feedback_input_class
        self.configuration = configparser.ConfigParser()
        self.load_configuration(configuration_path)

    def load_configuration(self, configuration_path: str) -> None:
        """
        Loads configuration in `configuration_path`
        """
        if configuration_path is not None:
            self.configuration.read(configuration_path)

    def build_api(self, kwargs: dict = None):
        """
        This function actually defines endpoint for our API, namely the
        `/predict` endpoint and the `/feedback` endpoint.
        """
        # retrieve fastapi configuration
        fastapi_configuration = (
            self.configuration['fastapi']
            if 'fastapi' in self.configuration.sections() else {}
        )

        # to override parameters in configuration file
        if kwargs is not None and 'fastapi' in kwargs.keys():
            fastapi_configuration.update(kwargs.get('fastapi'))

        app = FastAPI(**fastapi_configuration)

        # adding a route for predict
        predict_input_class = self.predict_input_class

        def _generate_request_uuid() -> UUID:
            """
            Helper function to generate a unique request_id to have a unique
            identifier of the request made.
            """
            return uuid4()

        @app.post("/predict")
        async def predict(input: predict_input_class) -> dict:
            request_uuid = _generate_request_uuid()
            return self.model.predict(input, request_uuid)

        # adding a route for feedback
        feedback_input_class = self.feedback_input_class

        @app.post("/feedback")
        async def feedback(input: feedback_input_class) -> dict:
            request_uuid = _generate_request_uuid()
            return {"status": "OK", 'request_uuid': request_uuid}

        return app
