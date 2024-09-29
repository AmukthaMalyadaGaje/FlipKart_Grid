from fastapi import APIRouter, HTTPException
from models.response_models import ExpiryResponse
from services.expiry_service import extract_expiry_date

router = APIRouter()


@router.post("/expiry-extraction", response_model=ExpiryResponse)
async def expiry_extraction(image_url: str):
    try:
        expiry_date = await extract_expiry_date(image_url)
        return {"expiry_date": expiry_date}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
