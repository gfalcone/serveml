import configparser
import logging

from uuid import uuid4

from fastapi import FastAPI

from serveml.predictions import AbstractPrediction
from serveml.inputs import BasicFeedbackInput
from serveml.outputs import BasicFeedbackOutput


class ApiBuilder(object):
    """
    Main class for generating an API thanks to FastAPI and Pydantic.
    """

    def __init__(
        self,
        model: AbstractPrediction,
        predict_input_class,
        predict_output_class=None,
        feedback_input_class=BasicFeedbackInput,
        feedback_output_class=None,
        configuration_path: str = None,
        api_prefix: str = "",
    ) -> None:
        """
        :param model: <serveml.ml.model.AbstractModel> object that inplements
        helper functions to have a proper API working.
        :param predict_input_class: <pydantic.BaseModel> object that implements the
        `/predict` input validator
        """
        self.model = model
        self.predict_input_class = predict_input_class
        self.predict_output_class = predict_output_class
        self.feedback_input_class = feedback_input_class
        self.feedback_output_class = feedback_output_class
        self.configuration = configparser.ConfigParser()
        self.load_configuration(configuration_path)
        self.api_prefix = api_prefix

    def load_configuration(self, configuration_path: str) -> None:
        """
        Loads configuration in `configuration_path`
        """
        if configuration_path is not None:
            self.configuration.read(configuration_path)

    @staticmethod
    def _generate_request_uuid() -> str:
        """
        Helper function to generate a unique request_id to have a unique
        identifier of the request made.
        """
        return str(uuid4())

    def _register_predict_endpoint(self, app: FastAPI):
        """
        This function will register the `/predict` endpoint on our API
        """
        predict_input_class = self.predict_input_class
        predict_output_class = self.predict_output_class

        @app.post("/predict", response_model=predict_output_class)
        async def predict(input: predict_input_class) -> dict:
            request_uuid = self._generate_request_uuid()
            logging.info("Begin prediction for: {}".format(request_uuid))
            prediction_result = self.model.predict(input, request_uuid)
            logging.info("Prediction done for: {}".format(request_uuid))
            return prediction_result

    def _register_feedback_endpoint(self, app: FastAPI):
        """
        This function will register the `/feedback` endpoint on our API
        """
        feedback_input_class = self.feedback_input_class
        feedback_output_class = self.feedback_output_class

        @app.post("/feedback", response_model=feedback_output_class)
        async def feedback(input: feedback_input_class) -> dict:
            request_uuid = self._generate_request_uuid()
            return {"status": True, "request_id": request_uuid}

    def build_api(self, kwargs: dict = None):
        """
        This function actually defines endpoint for our API, namely the
        `/predict` endpoint and the `/feedback` endpoint.
        """
        # retrieve fastapi configuration
        fastapi_configuration = (
            self.configuration["fastapi"]
            if "fastapi" in self.configuration.sections()
            else {}
        )

        # to override parameters in configuration file
        if kwargs is not None and "fastapi" in kwargs.keys():
            fastapi_configuration.update(kwargs.get("fastapi"))

        app = FastAPI(openapi_prefix=self.api_prefix, **fastapi_configuration)

        # registering endpoints
        self._register_predict_endpoint(app)
        self._register_feedback_endpoint(app)

        # handle the prefix
        if self.api_prefix != "":
            main_app = FastAPI()
            main_app.mount(self.api_prefix, app)
        else:
            main_app = app

        return main_app
