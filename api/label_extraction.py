from fastapi import APIRouter, HTTPException, UploadFile, File
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

router = APIRouter()

# Configure the path to the Tesseract executable if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image: Image.Image) -> np.ndarray:
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


def extract_labels(image: np.ndarray) -> dict:
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(image, config=custom_config)

    # Initialize a dictionary to hold extracted details
    labels = {
        "product_name": None,
        "company_name": None,
        "texture": None,
        "color": None,
        "size": None,
        "weight": None,
        "expiry_date": None,
        "price": None,
        "other_details": []
    }

    # Use regular expressions to match patterns
    for line in extracted_text.split('\n'):
        line = line.strip()
        if line:
            # Regex patterns for extracting various details
            if re.search(r'color[:\s]*([\w\s]+)', line, re.IGNORECASE):
                labels["color"] = re.search(
                    r'color[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()
            elif re.search(r'texture[:\s]*([\w\s]+)', line, re.IGNORECASE):
                labels["texture"] = re.search(
                    r'texture[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()
            elif re.search(r'size[:\s]*([\w\s]+)', line, re.IGNORECASE):
                labels["size"] = re.search(
                    r'size[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()
            elif re.search(r'weight[:\s]*([\w\s]+)', line, re.IGNORECASE):
                labels["weight"] = re.search(
                    r'weight[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()
            elif re.search(r'expiry[:\s]*([\w\s]+)', line, re.IGNORECASE):
                labels["expiry_date"] = re.search(
                    r'expiry[:\s]*([\w\s]+)', line, re.IGNORECASE).group(1).strip()
            elif re.search(r'price[:\s]*([\d.]+)', line, re.IGNORECASE):
                labels["price"] = re.search(
                    r'price[:\s]*([\d.]+)', line, re.IGNORECASE).group(1).strip()
            # Assuming the first valid line is the company name
            elif labels["company_name"] is None:
                labels["company_name"] = line
            elif labels["product_name"] is None:  # Assuming the next line is the product name
                labels["product_name"] = line
            else:
                labels["other_details"].append(line)

    return labels


@router.post("/label-extraction")
async def label_extraction(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image)
        labels = extract_labels(processed_image)
        return {"filename": file.filename, "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
