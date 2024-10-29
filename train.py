import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define paths and hyperparameters
dataset_dir = 'dataset'  # Update this to your dataset path
image_size = (224, 224)
batch_size = 32
epochs_top_level = 10
epochs_sub_level = 10

# First Stage: Top-level classification (e.g., 'apples', 'bananas')

# Data generators for the first level of classification
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load a pre-trained model (ResNet50) without the top layers
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

# Add new layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
top_level_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
top_level_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy', metrics=['accuracy'])

# Train the top-level classifier
top_level_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs_top_level
)

# Save the top-level model
top_level_model.save('top_level_classifier.h5')


# Second Stage: Train a subcategory classifier for the top-level category predicted

def train_subcategory_model(top_category):
    # Subdirectory for the top-level category
    subdir = os.path.join(dataset_dir, top_category)

    # Create data generators for the subdirectory
    train_subdir_gen = train_datagen.flow_from_directory(
        subdir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_subdir_gen = train_datagen.flow_from_directory(
        subdir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Load a new ResNet50 model for subcategory classification
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=(224, 224, 3))

    # Add classification layers for subcategories
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    sub_predictions = Dense(train_subdir_gen.num_classes,
                            activation='softmax')(x)

    # Create the subcategory classifier model
    sub_model = Model(inputs=base_model.input, outputs=sub_predictions)

    # Freeze the base model's layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the subcategory model
    sub_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the subcategory classifier
    sub_model.fit(
        train_subdir_gen,
        validation_data=validation_subdir_gen,
        epochs=epochs_sub_level
    )

    # Save the subcategory model
    sub_model.save('subcategory_classifier.h5')

# Call this function once the top-level category is predicted to train the subcategory model
