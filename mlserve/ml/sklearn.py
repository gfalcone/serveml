import pandas as pd
import numpy as np

from pydantic import BaseModel

from mlserve.ml import AbstractModel
from mlserve.io import pandas_to_dict, pydantic_model_to_pandas


class SklearnModel(AbstractModel):
    """
    Implementation of <mlserve.ml.model.AbstractModel> for scikit-learn
    """
    def _transform_input(self, input: BaseModel):
        """
        Transforms <pydantic.BaseModel> object to <pandas.DataFrame>
        """
        return pydantic_model_to_pandas(input)

    def _apply_model(self, transformed_input: pd.DataFrame):
        """
        Applies the sklearn model to the <pandas.DataFrame>.
        Returns either one of these:
            - <pandas.DataFrame>
            - <pandas.Series>
            - <numpy.ndarray>
        """
        return self.model.predict(transformed_input)

    def _transform_output(self, output):
        """
        Transforms output given by <mlserve.ml.sklearn._apply_model> to
        prepare sending result with API.
        """
        if isinstance(output, np.ndarray):
            result = output.tolist()
        elif isinstance(output, pd.DataFrame):
            result = pandas_to_dict(output)
        elif isinstance(output, pd.Series):
            result = output.to_dict()
        return {'result': result}
