from fastapi import APIRouter, HTTPException, UploadFile, File
from models.response_models import ExpiryResponse
import re
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print("Expiry extraction function called.")  # Simple print for debugging

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

router = APIRouter()

def extract_expiry_date_from_text(text: str) -> str:
    date_patterns = [
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
        r'\b(\d{1,2} \w{3} \d{2,4})\b',
        r'\b(\w{3} \d{1,2}, \d{2,4})\b',
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',
        r'\b(\d{1,2}(st|nd|rd|th) [A-Za-z]+ \d{4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            print(f"Matched expiry date pattern: {match.group(0)}")  # Debug print
            return match.group(0)

    print("No expiry date found in text.")  # Debug print
    return "No expiry date found."

async def extract_expiry_date(image_file: UploadFile) -> str:
    try:
        image = Image.open(image_file.file).convert("RGB")
        logger.info("Image loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        image_np = np.array(image)
        result = ocr.ocr(image_np, cls=True)
        logger.info("OCR processing completed successfully.")
        print(f"OCR Result: {result}")  # Debug print
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    extracted_text = " ".join(word_info[1][0] for line in result for word_info in line)
    logger.info(f"Extracted text: {extracted_text}")
    print(f"Extracted Text: {extracted_text}")  # Debug print

    expiry_date = extract_expiry_date_from_text(extracted_text)
    return expiry_date

@router.post("/extract-expiry/", response_model=ExpiryResponse)
async def expiry_extraction(image_file: UploadFile = File(...)):  # Ensure File(...) is used
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    expiry_date = await extract_expiry_date(image_file)
    print(f"Extracted Expiry Date: {expiry_date}")  # Debug print
    return ExpiryResponse(expiry_date=expiry_date)
