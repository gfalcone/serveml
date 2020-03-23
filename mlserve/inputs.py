from pydantic import BaseModel


class FeedbackInput(BaseModel):
    """
    Basic input validator for feedbacks
    """
    request_id: str
    status: bool
    expected_result: str = None
