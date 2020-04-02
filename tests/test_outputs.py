import unittest

from pydantic.error_wrappers import ValidationError

from serveml.outputs import BasicFeedbackOutput


class TestOutputs(unittest.TestCase):
    def test_parsing_good_output(self):
        feedback_input = {"request_id": "coconut", "status": True}
        feedback_object = BasicFeedbackOutput(**feedback_input)
        self.assertEqual(feedback_object.request_id, "coconut")
        self.assertEqual(feedback_object.status, True)

    def test_parsing_output_with_wrong_type(self):
        feedback_input = {"request_id": "coconut", "status": "coconut"}
        with self.assertRaises(ValidationError):
            BasicFeedbackOutput(**feedback_input)

    def test_parsing_output_with_wrong_additional_inputs(self):
        feedback_input = {
            "request_id": "coconut",
            "status": True,
            "awesome": True,
        }
        feedback = BasicFeedbackOutput(**feedback_input)
        expected_keys = {"request_id", "status"}
        self.assertEqual(expected_keys, feedback.dict().keys())
