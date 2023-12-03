from fastapi.routing import APIRouter
from app.models import ModelInferenceRequest, ModelInferenceResponse
from app.logic import predict

router = APIRouter()


@router.post("/predict", tags=["Model inference"], response_model=ModelInferenceResponse)
async def inference(request: ModelInferenceRequest):
    prediction = predict(request.type, request.image_path)
    return ModelInferenceResponse(prediction=prediction)
