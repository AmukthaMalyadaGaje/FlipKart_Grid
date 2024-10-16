import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class BrandService:
    def __init__(self):
        # Load your trained model
        self.model = load_model('brand_recognition_model.h5')
        self.class_names = self.load_class_names()  # Load class names

    def load_class_names(self):
        class_names = []
        try:
            with open('brand_mapping.csv', 'r', encoding='utf-8') as f:  # Specify utf-8 encoding
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 1:  # Ensure there is at least one part
                        class_names.append(parts[0])  # Append logo name
        except FileNotFoundError:
            logging.error("Class names file not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading class names: {e}")
        return class_names

    def recognize_brand(self, file):
        try:
            # Use a temporary file for the uploaded image
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.file.read())
                img_path = temp_file.name  # Get the temporary file path

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(
                150, 150))  # Adjust target size
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(
                img_array, axis=0) / 255.0  # Normalize the image

            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[
                0]  # Get class index
            confidence = np.max(predictions)  # Get confidence score

            # Remove the temporary file after processing
            os.remove(img_path)

            # Convert confidence to float for JSON serialization
            confidence = float(confidence)

            # Return the brand name and confidence
            brand_name = self.class_names[predicted_class]
            return brand_name, confidence

        except Exception as e:
            logging.error(f"Error during brand recognition: {e}")
            return None, None  # Return None or raise an exception
