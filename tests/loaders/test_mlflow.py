import unittest

from mlserve.loaders.mlflow import MlflowModelLoader


class TestMlflowModelLoader(unittest.TestCase):
    def setUp(self):
        self.mlflowModelLoader = MlflowModelLoader('http://localhost:5000')

    def test_load_model_with_right(self):
        pass
