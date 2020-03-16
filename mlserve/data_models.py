from pydantic import BaseModel


class FeedbackModel(BaseModel):
    request_id: str
    status: bool
    expected_result: str = None
