# services/brand_service.py
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class BrandRecognitionService:
    def __init__(self, model_path, brand_images_dir):
        self.model = load_model(model_path)
        self.brand_images_dir = brand_images_dir
        self.label_encoder = self.load_label_encoder()

    def load_label_encoder(self):
        brands = os.listdir(self.brand_images_dir)
        return LabelEncoder().fit(brands)

    def predict_brand(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "Image not found"

        image = cv2.resize(image, (64, 64))
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize

        predictions = self.model.predict(image)
        predicted_class = self.label_encoder.inverse_transform(
            [np.argmax(predictions)])
        return predicted_class[0]
