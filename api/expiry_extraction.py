from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import re
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import logging
from datetime import datetime
import cv2

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

router = APIRouter()

class DatesResponse(BaseModel):
    highest_date: str
    raw_text: str
    extracted_dates: list

def normalize_date(date_str: str) -> datetime:
    date_formats = [
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%b %d, %Y", "%B %d, %Y", "%d%b%Y", "%d%B%Y",
        "%d %b%Y", "%d%b%Y", "%d%m%Y"
    ]
    
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str.strip(), date_format)
        except ValueError:
            continue
    return None

def extract_dates_from_text(text: str) -> list:
    date_patterns = [
        r'(\d{1,2}\s?[/-]?\s?[A-Za-z]{3,}\s?[/-]?\s?\d{4})',
        r'([A-Za-z]{3,}\s?\d{1,2},?\s?\d{4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}\s?[A-Za-z]+\s?\d{2,4})',
        r'(\d{1,2}[A-Za-z]+\d{4})',
        r'(\d{8})',
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
    ]
    
    all_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        all_dates.extend(matches)
    
    normalized_dates = []
    for date_str in all_dates:
        normalized_date = normalize_date(date_str)
        if normalized_date:
            normalized_dates.append(normalized_date)
    
    return normalized_dates

def preprocess_image(image: Image) -> np.ndarray:
    # Convert to OpenCV format (BGR)
    image_cv = np.array(image)[:, :, ::-1]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    return binary_image

async def extract_highest_valid_date(image_file: UploadFile) -> DatesResponse:
    try:
        image = Image.open(image_file.file).convert("RGB")
        logger.info("Image loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    try:
        # Perform OCR on the preprocessed image
        result = ocr.ocr(preprocessed_image, cls=True)
        logger.info("OCR processing completed successfully.")
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    extracted_text = " ".join(word_info[1][0] for line in result for word_info in line)
    logger.info(f"Extracted text: {extracted_text}")

    extracted_dates = extract_dates_from_text(extracted_text)
    current_year = datetime.now().year
    valid_dates = [date for date in extracted_dates if isinstance(date, datetime) and current_year <= date.year <= current_year + 30]

    if not valid_dates:
        logger.info("No valid dates found within the 30-year range.")
        highest_date = "No valid date found"
    else:
        highest_date = max(valid_dates).strftime("%d-%b-%Y")
        logger.info(f"Highest valid date: {highest_date}")

    formatted_extracted_dates = [date.strftime("%d-%b-%Y") for date in extracted_dates]

    return DatesResponse(highest_date=highest_date, raw_text=extracted_text, extracted_dates=formatted_extracted_dates)

@router.post("/expiry-extraction", response_model=DatesResponse)
async def date_extraction(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")

    if not file.content_type.startswith('image/'):
        logger.error("Invalid file type. Only images are allowed.")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    response = await extract_highest_valid_date(file)
    logger.info(f"Highest Date: {response.highest_date}")
    
    return response
