# from fastapi import APIRouter, File, UploadFile, HTTPException
# from services.count_service import count_items
# from models.response_models import CountResponse

# router = APIRouter()


# @router.post("/count-items", response_model=CountResponse)
# async def count_items(image_visible: UploadFile = File(...), image_ir: UploadFile = File(...)):
#     if image_visible.content_type not in ["image/jpeg", "image/png"] or image_ir.content_type not in ["image/jpeg", "image/png"]:
#         raise HTTPException(status_code=400, detail="Invalid image type")

#     count = count_items(image_visible, image_ir)

#     return {"count": count}
