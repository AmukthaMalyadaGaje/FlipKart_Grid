import numpy as np
from PIL import Image
from api.unet_model import unet_model

# Load the U-Net model (ensure to load a trained model if available)
model = unet_model()  # or load a pre-trained model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert image to a numpy array and preprocess for U-Net.
    """
    img_array = np.array(image.resize((256, 256))
                         )  # Resize to match model input size
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


def remove_background_with_unet(image: Image.Image) -> Image.Image:
    """
    Use U-Net to remove the background from the input image.
    """
    img_array = preprocess_image(image)

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with U-Net
    # Predict the mask (output shape: (1, 256, 256, 1))
    mask = model.predict(img_array)

    # Check if mask is None or has incorrect shape
    if mask is None or mask.shape != (1, 256, 256, 1):
        raise ValueError(
            "The U-Net model returned None or an unexpected mask shape.")

    # Post-process the mask
    mask = (mask[0] > 0.5).astype(np.uint8)  # Convert to binary mask

    # Resize the mask to original image size
    mask_resized = Image.fromarray(
        mask * 255).resize(image.size, Image.BILINEAR)  # Resize to original size

    # Convert mask to a 3-channel image for multiplication
    mask_resized = mask_resized.convert("L")  # Convert mask to grayscale
    # Convert to array and normalize
    mask_resized = np.array(mask_resized) / 255.0

    # Ensure the mask has the same number of channels as the original image
    if len(mask_resized.shape) == 2:  # If mask is 2D, add a channel dimension
        mask_resized = np.expand_dims(mask_resized, axis=-1)

    # Remove the background from the original image
    img_with_bg_removed = img_array[0] * mask_resized  # Apply the mask

    # Ensure the resulting image has the correct shape (256, 256, 3)
    # Scale back to [0, 255]
    img_with_bg_removed = (img_with_bg_removed * 255).astype(np.uint8)

    # Convert back to Image
    img_with_bg_removed = Image.fromarray(img_with_bg_removed)

    return img_with_bg_removed
