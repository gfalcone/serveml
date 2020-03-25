from typing import List

import pandas as pd

from keras.preprocessing.text import Tokenizer

from serveml.api import ApiBuilder
from serveml.inputs import BasicInput
from serveml.loader import load_mlflow_model
from serveml.predictions import GenericPrediction

# load model
model = load_mlflow_model(
    # MlFlow model path
    "models:/keras_model/1",
    # MlFlow Tracking URI
    "http://localhost:5000",
)


# Implement deserializer for input data
class ReutersNewswireTopic(BasicInput):
    sequence: List[int]


# Implement prediction because this is a bit custom
class CustomKerasApplication(GenericPrediction):
    def _transform_input(self, input: ReutersNewswireTopic):
        """
        Transforms <ReutersNewswireTopic> object to <numpy.array>
        """
        max_words = 1000
        tokenizer = Tokenizer(num_words=max_words)
        x_train = tokenizer.sequences_to_matrix(
            [input.sequence], mode="binary"
        )
        return pd.DataFrame(x_train)


# implement application
app = ApiBuilder(
    CustomKerasApplication(model), ReutersNewswireTopic
).build_api()
