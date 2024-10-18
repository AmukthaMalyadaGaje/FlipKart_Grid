import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.freshness_service import FreshnessService
from PIL import Image
import numpy as np
from io import BytesIO  # Import BytesIO for conversion
# Import the background removal function
from api.background_removal import remove_background

router = APIRouter()
freshness_service = FreshnessService(
    'C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\shelf_life_prediction\\best_model.keras'
)


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = './temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file temporarily
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Load the image for background removal
        original_image = Image.open(file_location).convert("RGB")

        # Save the original image to a JPEG bytes-like object
        img_byte_arr = BytesIO()
        original_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)  # Seek to the beginning of the BytesIO object

        # Now call the remove_background function with the JPEG bytes
        image_with_bg_removed = remove_background(
            img_byte_arr.getvalue())  # Pass bytes directly

        # Load the JPEG bytes into a NumPy array
        processed_image = Image.open(BytesIO(image_with_bg_removed))

        # Save processed image to a temporary file for prediction
        processed_image_location = os.path.join(
            temp_dir, "processed_" + file.filename)
        processed_image.save(processed_image_location)

        # Predict the shelf life using the processed image
        predicted_shelf_life = freshness_service.predict_shelf_life(
            processed_image_location)

        # Clean up: remove the saved files
        os.remove(file_location)
        os.remove(processed_image_location)

        return JSONResponse(content={"predicted_shelf_life": predicted_shelf_life})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
