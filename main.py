from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.label_extraction import router as label_router
from api.freshness_prediction import router as freshness_router
# from api.brand_recognition import router as brand_router
from api.expiry_extraction import router as expiry_router
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Allows requests from this origin
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Include routers
# app.include_router(brand_router)
app.include_router(label_router)
app.include_router(expiry_router)
app.include_router(freshness_router)
# Added a prefix to match the router in label_extraction
app.include_router(label_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the expiry extraction API!"}
