from fastapi import APIRouter, File, UploadFile
from services.brand_service import BrandService

router = APIRouter()
brand_service = BrandService()


@router.post("/recognize-brand")
async def recognize_brand(file: UploadFile = File(...)):
    brand_name, confidence = brand_service.recognize_brand(file)
    if brand_name is not None:
        return {"brand": brand_name, "confidence": confidence}
    else:
        return {"error": "Brand recognition failed."}
