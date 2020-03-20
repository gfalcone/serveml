import unittest

import numpy as np
import pandas as pd

from mlserve.data_models import BaseModel, FeedbackModel
from mlserve.io import dict_to_pandas
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction


class TestGenericPrediction(unittest.TestCase):
    def setUp(self):
        self.model = load_mlflow_model(
            'models:/sklearn_model/1', 'http://localhost:5000'
        )
        self.prediction = GenericPrediction(self.model)

    def test__fetch_data(self):
        self.assertIsNone(self.prediction._fetch_data(BaseModel()))

    def test__combine_fetched_data_with_input(self):
        transformed_input = dict_to_pandas({'item_id': 0, 'name': 'coconut'})
        self.assertTrue(
            self.prediction._combine_fetched_data_with_input(
                None,
                transformed_input
            ).equals(transformed_input)
        )

    def test__generate_request_uuid(self):
        result = self.prediction._generate_request_uuid({'item_id': 0})
        self.assertIsNotNone(result['request_id'])

    def test__transform_input(self):
        item = {
            'request_id': 'coconut',
            'status': True,
            'expected_result': 'coco'
        }
        result = self.prediction._transform_input(FeedbackModel(**item))
        self.assertTrue(result.equals(dict_to_pandas(item)))

    def test__apply_model(self):
        item = {
            'alcohol': 0.0,
            'chlorides': 0.0,
            'citric_acid': 0.0,
            'density': 0.0,
            'fixed_acidity': 0.0,
            'free_sulfur_dioxide': 0,
            'pH': 0.0,
            'residual_sugar': 0.0,
            'sulphates': 0.0,
            'total_sulfur_dioxide': 0,
            'volatile_acidity': 0,
        }
        df = dict_to_pandas(item)
        self.model.predict(df)

    def test__transform_output(self):
        items_array = [1, 2, 3]
        self.assertEqual(
            {'result': items_array},
            self.prediction._transform_output(np.array(items_array))
        )
        items_dict = {
            'item_id': 0,
            'name': 'coconut'
        }
        items_df = dict_to_pandas(items_dict)
        self.assertEqual(
            {'result': [items_dict]},
            self.prediction._transform_output(items_df)
        )
        data = np.array(['a', 'b', 'c', 'd'])
        s = pd.Series(data)

        self.assertEqual(
            {
                'result': {
                    0: 'a',
                    1: 'b',
                    2: 'c',
                    3: 'd'
                }
            },
            self.prediction._transform_output(s)
        )

        self.assertEqual(
            {'result': None},
            self.prediction._transform_output('it')
        )
