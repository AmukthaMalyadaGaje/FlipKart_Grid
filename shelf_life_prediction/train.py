import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import create_model

# Set paths for training and testing data
train_data_dir = 'path/to/train/directory'  # Update with your train directory
test_data_dir = 'path/to/test/directory'      # Update with your test directory

# Define image dimensions
input_shape = (150, 150, 3)

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

# Get the number of classes from the training data
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Create the model with both input_shape and num_classes
model = create_model(input_shape, num_classes)

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)
