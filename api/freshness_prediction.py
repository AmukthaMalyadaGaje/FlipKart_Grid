# api/freshness_extraction.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.freshness_service import FreshnessService
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

router = APIRouter()
# freshness_service = FreshnessService('shelf_life_model.h5')
freshness_service = FreshnessService(
    'C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\shelf_life_prediction\\shelf_life_model.h5')


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())

        # Load the image for display
        original_image = Image.open(file.filename)
        img_array = np.array(original_image)

        # Normalize the image
        normalized_img_array = img_array / 255.0

        # Display the original and normalized images
        plt.figure(figsize=(10, 5))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        # Normalized Image
        plt.subplot(1, 2, 2)
        plt.imshow(normalized_img_array)
        plt.title("Normalized Image")
        plt.axis("off")

        plt.show()  # Show the images

        # Predict the shelf life
        predicted_shelf_life = freshness_service.predict_shelf_life(
            file.filename)

        # Clean up: remove the saved file
        os.remove(file.filename)

        return JSONResponse(content={"predicted_shelf_life": predicted_shelf_life})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
