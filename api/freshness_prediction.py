from fastapi import APIRouter, HTTPException
from models.response_models import FreshnessResponse
from services.freshness_service import predict_freshness

router = APIRouter()


@router.post("/freshness-prediction", response_model=FreshnessResponse)
async def freshness_prediction(image_url: str):
    try:
        freshness_score = await predict_freshness(image_url)
        return {"freshness_score": freshness_score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
