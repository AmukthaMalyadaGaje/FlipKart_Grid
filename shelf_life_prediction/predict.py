import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def predict_shelf_life(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class[0]  # Adjust based on your class mapping

if _name_ == "_main_":
    model = load_model('shelf_life_model.h5')
    new_image_path = 'path_to_new_image.jpg'  # Update with the actual path
    predicted_shelf_life = predict_shelf_life(new_image_path, model)
    print(f"Predicted Shelf Life: {predicted_shelf_life}")