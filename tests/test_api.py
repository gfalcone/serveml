import unittest

from mlserve.api import ApiBuilder
from mlserve.loader import load_mlflow_model
from mlserve.predictions import GenericPrediction
from examples.serving.sklearn import WineComposition


class TestApiBuilder(unittest.TestCase):
    def setUp(self):
        model = load_mlflow_model(
            # MlFlow model path
            'models:/sklearn_model/1',
            # MlFlow Tracking URI
            'http://localhost:5000',
        )
        self.api_builder = ApiBuilder(
            model=GenericPrediction(model),
            predict_input_class=WineComposition,
            configuration_path='api.cfg'
        )

    def test_load_configuration(self):
        fast_api_configuration = {
            'title': 'WineQualityApi',
            'description': 'This API helps you determine if the quality of the Wine is good or not',  # NOQA
            'version': '0.1.0'
        }
        self.assertEqual(
            self.api_builder.configuration['fastapi'],
            fast_api_configuration
        )

    def test_build_api(self):
        app = self.api_builder.build_api()
        paths = list(map(lambda x: x.path, app.router.routes))
        self.assertIn('/predict', paths)
        self.assertIn('/feedback', paths)
        self.assertIn('/docs', paths)
        self.assertIn('/redoc', paths)
