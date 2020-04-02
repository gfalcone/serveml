from pydantic import BaseModel as BasicInput


class BasicPredictOutput(BasicInput):
    """
    Basic input validator for feedbacks
    """

    request_id: str


class BasicFeedbackOutput(BasicInput):
    """
    Basic input validator for feedbacks
    """

    status: bool
    request_id: str
