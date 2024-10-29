from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import re
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

router = APIRouter()

# Define Pydantic models


class LabelResponse(BaseModel):
    label: str


class ExpiryResponse(BaseModel):
    expiry_date: str
    raw_text: str


class BrandRecognitionResponse(BaseModel):
    brand: str


class CountResponse(BaseModel):
    count: int


class FreshnessResponse(BaseModel):
    freshness_score: float


class BrandResponseModel(BaseModel):
    brand: str


class DatesResponse(BaseModel):
    highest_date: str
    expiry_date: str
    raw_text: str
    extracted_dates: list


def normalize_date(date_str: str) -> datetime:
    date_formats = [
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%b %d, %Y", "%B %d, %Y", "%d%b%Y", "%d%B%Y",
        "%d %b%Y", "%d%b%Y", "%d%m%Y", "%d%m%y"
    ]

    # Handle ddmmyy (like 040917 -> 04-09-2017)
    if len(date_str) == 6:
        try:
            return datetime.strptime(date_str, "%d%m%y")
        except ValueError:
            pass

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str.strip(), date_format)
        except ValueError:
            continue
    return None


def extract_dates_from_text(text: str) -> dict:
    date_patterns = [
        r'(MFG[:\- ]?\d{6})',        # MFG ddmmyy
        r'(EXP[:\- ]?\d{6})',        # EXP ddmmyy
        r'(\d{1,2}\s?[/-]?\s?[A-Za-z]{3,}\s?[/-]?\s?\d{4})',  # dd Mon yyyy
        r'([A-Za-z]{3,}\s?\d{1,2},?\s?\d{4})',                # Mon dd, yyyy
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',                     # yyyy-mm-dd
        # dd-mm-yyyy or dd-mm-yy
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        # ddmmyyyy or ddmmyy
        r'(\d{8})'
    ]

    expiry_dates = []
    other_dates = []

    for pattern in date_patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            if "EXP" in match.upper():
                date_str = re.sub(r'EXP[:\- ]?', '',
                                  match, flags=re.IGNORECASE)
                date = normalize_date(date_str)
                if date:
                    expiry_dates.append(date)
            elif "MFG" in match.upper():
                date_str = re.sub(r'MFG[:\- ]?', '',
                                  match, flags=re.IGNORECASE)
                date = normalize_date(date_str)
                if date:
                    other_dates.append(date)
            else:
                date = normalize_date(match)
                if date:
                    other_dates.append(date)

    return {
        "expiry_dates": expiry_dates,
        "other_dates": other_dates
    }


async def extract_highest_valid_date(image_file: UploadFile) -> DatesResponse:
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
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    extracted_text = " ".join(word_info[1][0]
                              for line in result for word_info in line)
    logger.info(f"Extracted text: {extracted_text}")

    # Extract dates from the text
    dates_dict = extract_dates_from_text(extracted_text)

    current_year = datetime.now().year
    valid_expiry_dates = [date for date in dates_dict["expiry_dates"]
                          if current_year <= date.year <= current_year + 30]
    valid_other_dates = [date for date in dates_dict["other_dates"]
                         if current_year - 30 <= date.year <= current_year + 30]

    expiry_date = max(valid_expiry_dates) if valid_expiry_dates else None
    highest_date = expiry_date or max(
        valid_other_dates, default="No valid date found")

    expiry_date_str = expiry_date.strftime(
        "%d-%b-%Y") if expiry_date else "No expiry date found"
    formatted_extracted_dates = [date.strftime(
        "%d-%b-%Y") for date in valid_other_dates]

    return DatesResponse(
        highest_date=highest_date.strftime(
            "%d-%b-%Y") if isinstance(highest_date, datetime) else "No valid date found",
        expiry_date=expiry_date_str,
        raw_text=extracted_text,
        extracted_dates=formatted_extracted_dates
    )


@router.post("/expiry-extraction", response_model=DatesResponse)
async def date_extraction(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")

    if not file.content_type.startswith('image/'):
        logger.error("Invalid file type. Only images are allowed.")
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image.")

    response = await extract_highest_valid_date(file)
    logger.info(f"Expiry Date: {response.highest_date}")

    return response
