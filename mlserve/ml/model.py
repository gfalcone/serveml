from abc import ABC, abstractmethod
from uuid import uuid4


class AbstractModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _transform_input(self, input):
        pass

    @abstractmethod
    def _transform_output(self, output):
        pass

    @abstractmethod
    def _apply_model(self, transfomed_input):
        pass

    @staticmethod
    def _generate_request_uuid(transformed_output):
        transformed_output['request_id'] = uuid4()
        return transformed_output

    def predict(self, input):
        transformed_input = self._transform_input(input)
        output = self._apply_model(transformed_input)
        transformed_output = self._transform_output(output)
        final_output = self._generate_request_uuid(transformed_output)
        return final_output
