# train_model.py
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from dataset_preparation import load_data

# Load dataset
images, labels = load_data(
    'C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\brand_images\\logos')

# Preprocess the data
images = images.astype('float32') / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images, labels, epochs=20, batch_size=32, validation_split=0.2)

# Save the model and class labels
model.save('brand_recognition_model.h5')
np.save('classes.npy', label_encoder.classes_)
print("Model trained and saved.")
