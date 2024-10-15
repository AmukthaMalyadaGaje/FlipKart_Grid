# services/freshness_service.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class_mapping = {
    0: 'apples(1-5)',
    1: 'apples(6-10)',
    2: 'apples(11-15)',
    3: 'apples(16-20)',
    4: 'applesExpired',
    5: 'bananas(1-2)',
    6: 'bananas(3-4)',
    7: 'bananas(5-7)',
    8: 'bananas(8-10)',
    9: 'bananasexpired',
}


class FreshnessService:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict_shelf_life(self, img_path):
        img = image.load_img(img_path, target_size=(150, 150))  # Resize image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        return class_mapping[predicted_class[0]]
