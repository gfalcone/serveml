from abc import ABC, abstractmethod
from uuid import uuid4

from pydantic import BaseModel


class AbstractModel(ABC):
    """
    Abstract class to define methods called during predict
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _transform_input(self, input: BaseModel):
        """
        Function called right after API call. It is supposed to transform
        <pydantic.BaseModel> object into the input data format needed to apply
        model
        """

    @staticmethod
    def _fetch_data(input: BaseModel):
        """
        Helper function in case we need additional data. In most of the cases,
        can be ignored
        """
        pass

    @staticmethod
    def _combine_fetched_data_with_input(fetched_data, transformed_input):
        return transformed_input

    @abstractmethod
    def _apply_model(self, transformed_input):
        """
        Function called to apply Machine Learning model to predict from the
        transformed input
        """

    @abstractmethod
    def _transform_output(self, output):
        """
        Function called right after applying model to input data. Supposed to
        transform the data that we got after the predict in order to
        """

    @staticmethod
    def _generate_request_uuid(transformed_output):
        """
        Helper function to generate a unique request_id to have a unique
        identifier of the request made.
        """
        transformed_output['request_id'] = uuid4()
        return transformed_output

    def predict(self, input):
        """
        Main function that will be used by all the childs to apply model.
        Here are the steps made :
            - Transform <pydantic.BaseModel> objet to target input object
            before applying model
            - Apply model
            - Transform output into more suitable format for an API
            - Add an uuid to the request to track request made
        """
        transformed_input = self._transform_input(input)
        fetched_data = self._fetch_data(input)
        combined_data = self._combine_fetched_data_with_input(
            fetched_data, transformed_input
        )
        output = self._apply_model(combined_data)
        transformed_output = self._transform_output(output)
        final_output = self._generate_request_uuid(transformed_output)
        return final_output
