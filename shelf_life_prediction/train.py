# train.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import create_model

# Define paths
train_data_dir = 'dataset/train'
test_data_dir = 'dataset/test'

# Image dimensions
img_width, img_height = 150, 150

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Determine the number of classes
num_classes = len(train_generator.class_indices)

# Create and train the model
model = create_model((img_width, img_height, 3), num_classes)
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the model
model.save('shelf_life_model.h5')