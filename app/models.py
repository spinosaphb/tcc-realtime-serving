from pydantic import BaseModel


class ModelInferenceRequest(BaseModel):
    type: str
    image_path: str


class ModelInferenceResponse(BaseModel):
    prediction: float