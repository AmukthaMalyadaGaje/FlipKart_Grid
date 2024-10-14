from paddleocr import PaddleOCR
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
import re
from api.bg_removal_with_unet import remove_background_with_unet

router = APIRouter()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')


@router.post("/label-extraction")
async def label_extraction(file: UploadFile = File(...)):
    """
    API endpoint to extract labels from an uploaded image using PaddleOCR.
    """
    try:
        # Load the image and convert to RGB
        image = Image.open(file.file).convert("RGB")

        # Remove background before OCR
        image_with_bg_removed = remove_background_with_unet(image)

        # Preprocess the image for OCR
        np_image = np.array(image_with_bg_removed)

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
