import os
import unittest

from mlflow.models import Model

from mlflow.exceptions import RestException

from mlserve.loaders.mlflow import MlflowModelLoader


class TestMlflowModelLoader(unittest.TestCase):
    def setUp(self):
        self.loader = MlflowModelLoader('http://localhost:5000')

    def test_load_model_with_right_parameters(self):
        model_path = 'model'
        destination_path = '/tmp'
        model = self.loader.load_model(
            'my_model',
            model_path,
            destination_path
        )
        self.assertIn(model_path, os.listdir(destination_path))
        self.assertIsInstance(model, Model)

    def test_load_model_with_wrong_model_name(self):
        with self.assertRaises(RestException):
            self.loader.load_model('coconut', 'model', '/tmp')

    def test_load_model_with_wrong_model_directory(self):
        with self.assertRaises(ValueError):
            self.loader.load_model('my_model', 'coconut', '/tmp')