# services/brand_service.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class BrandRecognitionService:
    def __init__(self, model_path: str, brand_images_dir: str):
        self.model = tf.keras.models.load_model(model_path)
        self.brand_images_dir = brand_images_dir
        self.class_indices = self.load_class_indices()

    def load_class_indices(self):
        # Create a mapping of brand names to indices
        brands = os.listdir(self.brand_images_dir)
        return {idx: brand for idx, brand in enumerate(brands)}

    def predict_brand(self, img_path: str):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)

            preds = self.model.predict(img_array)
            class_idx = np.argmax(preds, axis=1)[0]

            # Return the brand name corresponding to the predicted index
            predicted_brand = self.class_indices.get(
                class_idx, "Unknown Brand")
            return predicted_brand
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {"error": str(e)}
