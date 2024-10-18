# api/brand_recognition.py
import os
from fastapi import APIRouter, File, UploadFile
from services.brand_service import BrandRecognitionService
from fastapi.responses import JSONResponse

router = APIRouter()
brand_recognition_service = BrandRecognitionService(
    model_path='C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\brandRecognition\\brand_recognition_model.h5',
    brand_images_dir='brand_images/logos/'
)


@router.post("/predict_brand/")
async def predict_brand(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f'temp_{file.filename}'
        with open(temp_file_path, 'wb') as f:
            f.write(await file.read())
        print("Hello")
        # Predict the brand using the uploaded image
        predicted_brand = brand_recognition_service.predict_brand(
            temp_file_path)

        # Remove the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"predicted_brand": predicted_brand})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
