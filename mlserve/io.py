import pandas as pd
import json


def json_to_pandas(json_input):
    return dict_to_pandas(json.loads(json_input))


def pydantic_model_to_pandas(pydantic_model_input):
    return dict_to_pandas(pydantic_model_input.dict())


def dict_to_pandas(dictionary_input):
    return pd.DataFrame.from_dict([dictionary_input])


def pandas_to_dict(df: pd.DataFrame):
    return df.to_dict()
