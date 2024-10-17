# predict.py
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


def load_model_and_label_encoder():
    model = load_model('brand_recognition_model.h5')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(
        'classes.npy', allow_pickle=True)  # Load classes
    return model, label_encoder


def predict_brand(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize

    predictions = model.predict(image)
    predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])

    return predicted_class[0]


if __name__ == "__main__":
    model, label_encoder = load_model_and_label_encoder()
    # Change this to your test image path
    test_image_path = 'path_to_test_image/test_image.jpg'
    brand = predict_brand(test_image_path, model, label_encoder)
    print(f"The predicted brand is: {brand}")
