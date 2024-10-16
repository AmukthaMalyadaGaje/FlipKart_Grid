import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Load brand image mapping from JSON file
with open('C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\brand_image_mapping.json', 'r') as json_file:
    brand_image_data = json.load(json_file)

# Prepare data
images = []
labels = []
for brand, image_files in brand_image_data.items():
    for image_file in image_files:
        img_path = os.path.join('dataset/brand_images/logos', image_file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((150, 150))  # Resize to match model input
        images.append(np.array(img) / 255.0)  # Normalize image
        labels.append(brand)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the CNN model


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Create and compile the model
num_classes = len(label_encoder.classes_)
model = create_model((150, 150, 3), num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=10, batch_size=32)

# Save the model
model.save('brand_recognition_model.keras')
