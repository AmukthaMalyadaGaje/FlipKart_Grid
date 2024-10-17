# dataset_preparation.py
import os
import numpy as np
import cv2


def load_data(brand_images_dir):
    images = []
    labels = []
    brands = os.listdir(brand_images_dir)

    for brand in brands:
        brand_path = os.path.join(brand_images_dir, brand)
        if os.path.isdir(brand_path):
            for image_file in os.listdir(brand_path):
                image_path = os.path.join(brand_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:  # Ensure image is loaded
                    # Resize to fit model input
                    image = cv2.resize(image, (64, 64))
                    images.append(image)
                    labels.append(brand)

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    images, labels = load_data('brand_images/')
    print(f"Loaded {len(images)} images with labels.")
