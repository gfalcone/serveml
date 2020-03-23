import pandas as pd
from mlserve.inputs import BasicInput


def pydantic_model_to_pandas(pydantic_model_input) -> pd.DataFrame:
    """
    Function that transforms <pydantic.BaseModel> child objects to
    <pandas.DataFrame> objects
    :param pydantic_model_input: Input validator for API
    """
    return dict_to_pandas(pydantic_model_input.dict())


def dict_to_pandas(dictionary_input: dict) -> pd.DataFrame:
    """
    Function that transforms a dictionary into a <pandas.DataFrame>
    :param dictionary_input: Python dictionary
    """
    return pd.DataFrame.from_dict([dictionary_input])


def pandas_to_dict(df: pd.DataFrame) -> dict:
    """
    Function that transforms a <pandas.DataFrame> object to a Python Dictionary
    :param df: <pd.DataFrame> object
    """
    return df.to_dict('records')
