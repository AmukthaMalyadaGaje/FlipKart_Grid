import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.freshness_service import FreshnessService
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO  # Import BytesIO for conversion
# Import the background removal function
from api.background_removal import remove_background

router = APIRouter()
freshness_service = FreshnessService(
    'C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\shelf_life_prediction\\shelf_life_model.h5'
)


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = './temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file temporarily
        # Use os.path.join for better path handling
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Load the image for background removal
        original_image = Image.open(file_location)

        # Save the original image to a JPEG bytes-like object
        img_byte_arr = BytesIO()
        # Convert to JPEG format
        original_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)  # Seek to the beginning of the BytesIO object

        # Convert BytesIO to bytes
        img_bytes = img_byte_arr.getvalue()  # Get the bytes from the BytesIO object

        # Call the remove_background function with the JPEG bytes
        image_with_bg_removed = remove_background(img_bytes)

        # Load the JPEG bytes into a NumPy array
        processed_image = Image.open(BytesIO(image_with_bg_removed))

        # Convert to NumPy array
        processed_image_array = np.array(processed_image)

        # Ensure the processed image is in the correct shape for the model
        if processed_image_array.ndim == 2:  # If the image is grayscale
            processed_image_array = np.stack(
                (processed_image_array,) * 3, axis=-1)  # Convert to 3 channels

        # Normalize the image
        normalized_img_array = processed_image_array / 255.0

        # Display the original and processed images
        plt.figure(figsize=(10, 5))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        # Processed Image
        plt.subplot(1, 2, 2)
        plt.imshow(normalized_img_array)
        plt.title("Processed Image (Background Removed)")
        plt.axis("off")

        plt.show()  # Show the images

        # Predict the shelf life using the processed image
        predicted_shelf_life = freshness_service.predict_shelf_life(
            normalized_img_array)

        # Clean up: remove the saved file
        os.remove(file_location)

        return JSONResponse(content={"predicted_shelf_life": predicted_shelf_life})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
