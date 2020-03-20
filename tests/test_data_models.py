import unittest

from pydantic.error_wrappers import ValidationError

from mlserve.data_models import FeedbackModel


class TestDataModels(unittest.TestCase):
    def test_parsing_good_input(self):
        feedback_input = {
            'request_id': 'coconut',
            'status': True
        }
        feedback_object = FeedbackModel(**feedback_input)
        self.assertEqual(feedback_object.request_id, 'coconut')
        self.assertEqual(feedback_object.status, True)

    def test_parsing_input_with_wrong_type(self):
        feedback_input = {
            'request_id': 'coconut',
            'status': 'coconut'
        }
        with self.assertRaises(ValidationError):
            FeedbackModel(**feedback_input)

    def test_parsing_input_with_wrong_additional_inputs(self):
        feedback_input = {
            'request_id': 'coconut',
            'status': True,
            'awesome': True
        }
        feedback = FeedbackModel(**feedback_input)
        expected_keys = {'request_id', 'status', 'expected_result'}
        self.assertEqual(expected_keys, feedback.dict().keys())
