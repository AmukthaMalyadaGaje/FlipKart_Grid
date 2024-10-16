# main.py
import io
import numpy as np
import cv2
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
import torch

# Initialize FastAPI app
router = APIRouter()

# Load the YOLOv5 model (you can use 'yolov5s.pt' or a custom model)
# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


@router.post("/detect_brands/")
async def detect_brands(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Perform inference
    results = model(img)

    # Process results
    brand_detections = []
    # xyxy format: [xmin, ymin, xmax, ymax, confidence, class]
    for *box, conf, cls in results.xyxy[0]:
        brand_detections.append({
            "brand": model.names[int(cls)],  # Get the class name
            "confidence": float(conf),  # Get the confidence score
            "box": [float(x) for x in box]  # Get the bounding box coordinates
        })

    return JSONResponse(content={"detections": brand_detections})
