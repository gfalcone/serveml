import unittest

import pandas as pd

from mlserve.inputs import FeedbackInput
from mlserve.utils import (
    dict_to_pandas,
    pandas_to_dict,
    pydantic_model_to_pandas,
)


class TestUtils(unittest.TestCase):
    def test_parsing_dict_to_pandas(self):
        item = {"item_id": 0, "item_name": "coconut"}
        df = dict_to_pandas(item)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_parsing_pandas_to_dict(self):
        item = {"item_id": 0, "item_name": "coconut"}
        df = dict_to_pandas(item)
        self.assertEqual(pandas_to_dict(df), [item])

    def test_pydantic_model_to_pandas(self):
        feedback = FeedbackInput(status=True, request_id="coconut")
        result = pydantic_model_to_pandas(feedback)
        item = {
            "request_id": "coconut",
            "status": True,
            "expected_result": None,
        }
        self.assertTrue(result.equals(dict_to_pandas(item)))
