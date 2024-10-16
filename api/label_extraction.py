import cv2
import numpy as np
import re
from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
# Import your background removal function
from api.background_removal import remove_background

router = APIRouter()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')


@router.post("/label-extraction")
async def label_extraction(file: UploadFile = File(...)):
    """
    API endpoint to extract labels from an uploaded image using PaddleOCR.
    """
    try:
        # Load the image as bytes
        img_bytes = await file.read()  # Read the file content as bytes

        # Print the first few bytes for debugging
        # Optional: Check if the bytes are read correctly
        print(img_bytes[:10])

        # Open the image from bytes
        image = Image.open(BytesIO(img_bytes)).convert("RGB")  # Open the image

        # Remove background before OCR
        # Ensure this function is defined correctly
        image_with_bg_removed = remove_background(img_bytes)

        # Convert the bytes back to a numpy array for OCR
        nparr = np.frombuffer(image_with_bg_removed, np.uint8)
        img_with_bg_removed = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Use PaddleOCR for text recognition
        result = ocr.ocr(img_with_bg_removed)

        # Initialize a list to store the recognized text
        recognized_text = []

        # Extract recognized text from result
        for line in result:
            for res in line:
                recognized_text.append(res[1][0])

        # Initialize a dictionary to hold extracted details
        labels = {
            "product_name": None,
            "company_name": None,
            "mrp": None,
            "expiry_date": None,
            "other_details": []
        }

        # Use regex to extract common fields from the recognized text
        for line in recognized_text:
            line = line.strip()
            if line:
                # Extract MRP
                if re.search(r'mrp[:\s]*([\d.,]+)', line, re.IGNORECASE):
                    labels["mrp"] = re.search(
                        r'mrp[:\s]*([\d.,]+)', line, re.IGNORECASE).group(1).strip()

                # Extract Expiry Date
                elif re.search(r'expiry[:\s]*([\w\s]+)', line, re.IGNORECASE):
                    labels["expiry_date"] = re.search(
                        r'expiry[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()

                # Extract Company Name
                elif labels["company_name"] is None:
                    labels["company_name"] = line

                # Extract Product Name
                elif labels["product_name"] is None:
                    labels["product_name"] = line

                # Add other details
                else:
                    labels["other_details"].append(line)

        # Plot original and background-removed images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Image with Background Removed")
        plt.imshow(cv2.cvtColor(img_with_bg_removed, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()  # Display the plot

        return {"filename": file.filename, "labels": labels}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
