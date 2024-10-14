import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from api.unet_model import unet_model


def remove_background(image_bytes):
    # Load the image from bytes
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Create a mask for the grabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle around the object of interest
    rect = (10, 10, img.shape[1]-10, img.shape[0]-10)

    # Apply the grabCut algorithm
    cv2.grabCut(img, mask, rect, bgd_model,
                fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask so that all foreground pixels are marked as 1 and background as 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    img_fg = img * mask2[:, :, np.newaxis]

    # Encode the image to send back in the response
    _, encoded_img = cv2.imencode('.jpg', img_fg)

    return encoded_img.tobytes()  # Return the image as bytes
