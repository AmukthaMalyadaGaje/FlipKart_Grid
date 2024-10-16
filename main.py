# main.py
from fastapi import FastAPI
from api.label_extraction import router as label_router
from api.freshness_prediction import router as freshness_router
# from api.yolo_model import router as yolo_router
from api.brand_recognition import router as brand_router
import os
import warnings
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific warnings globally
warnings.filterwarnings("ignore")

# Suppress PaddleOCR logs
logging.getLogger('absl').setLevel(logging.ERROR)

# Create the FastAPI application
app = FastAPI(title="Image Processing API")

# Include routers
# app.include_router(yolo_router, prefix="/yolo", tags=["YOLO"])
app.include_router(brand_router)
# Uncomment if count_recognition is defined
# app.include_router(count_recognition.router)
app.include_router(label_router)
# Uncomment if expiry_router is defined
# app.include_router(expiry_router, prefix="/expiry", tags=["Expiry"])
app.include_router(freshness_router, prefix="/freshness",
                   tags=["Freshness Prediction"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API!"}
