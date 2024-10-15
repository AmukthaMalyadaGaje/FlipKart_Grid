# main.py
from train import train_model

if __name__ == "__main__":
    train_model()

    # To predict a shelf life for a new image, uncomment the following lines:
    # model = load_model('shelf_life_model.h5')
    # new_image_path = 'path_to_new_image.jpg'  # Update with the actual path
    # predicted_shelf_life = predict_shelf_life(new_image_path, model)
    # print(f"Predicted Shelf Life: {predicted_shelf_life}")
