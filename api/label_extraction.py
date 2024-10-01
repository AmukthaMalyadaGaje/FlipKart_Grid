# label_extraction.py
import keras_ocr
import cv2

# Initialize the keras-ocr pipeline globally to avoid reinitializing it on every request
pipeline = keras_ocr.pipeline.Pipeline()

def extract_labels(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use the keras-ocr pipeline to extract text
    prediction_groups = pipeline.recognize([image])
    
    # Flatten predictions to get text labels
    labels = [text for text, _ in prediction_groups[0]]
    
    return labels
