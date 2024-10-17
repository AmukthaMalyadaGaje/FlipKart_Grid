from paddleocr import PaddleOCR
import numpy as np
import cv2  # For image preprocessing
from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
import re

router = APIRouter()  # Keep the router declaration here

# Initialize PaddleOCR with angle classification enabled
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert image to a numpy array and apply OpenCV preprocessing techniques to improve text extraction.
    """
    # Convert the PIL image to a NumPy array (OpenCV format: BGR)
    img_array = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to make text clearer
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Optionally, apply morphology operations (dilation/erosion) if text is too thin or too thick
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed_image


@router.post("/label-extraction/")
async def label_extraction(file: UploadFile = File(...)):
    """
    API endpoint to extract labels from an uploaded image using PaddleOCR.
    """
    try:
        # Load the image and convert to RGB
        image = Image.open(file.file).convert("RGB")

        # Preprocess the image using OpenCV
        np_image = preprocess_image(image)

        # Use PaddleOCR for text recognition
        result = ocr.ocr(np_image)

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

        return {"filename": file.filename, "labels": labels}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
