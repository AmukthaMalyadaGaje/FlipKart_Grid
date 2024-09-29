import numpy as np
import requests
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

app = FastAPI()


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return img


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize and preprocess the image for MobileNetV2
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


def decode_predictions(preds: np.ndarray) -> str:
    # Decode predictions to get label
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[
        0][0]
    return decoded[1]  # return the label (class name)


async def extract_label_from_image(image_url: str) -> str:
    try:
        # Load and preprocess the image
        image = load_image_from_url(image_url)
        img_array = preprocess_image(image)

        # Make predictions
        preds = model.predict(img_array)

        # Decode and return the label
        label = decode_predictions(preds)
        return label
    except Exception as e:
        raise Exception(f"Error in label extraction: {str(e)}")


@app.post("/label-extraction")
async def label_extraction(image_url: Optional[str] = None, file: UploadFile = File(None)):
    if image_url is None and file is None:
        raise HTTPException(
            status_code=400, detail="Either 'image_url' or 'file' must be provided")

    try:
        if image_url:
            # Extract label from the image URL
            label = await extract_label_from_image(image_url)
            return JSONResponse(content={"image_url": image_url, "label": label})

        if file:
            # Load image from uploaded file
            image = Image.open(file.file).convert("RGB")
            img_array = preprocess_image(image)

            # Make predictions
            preds = model.predict(img_array)

            # Decode and return the label
            label = decode_predictions(preds)
            return JSONResponse(content={"filename": file.filename, "label": label})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
