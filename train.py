import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

# Load brand mapping CSV file
df = pd.read_csv('brand_mapping.csv')

# Check for missing images and remove brands with missing images
image_files = set(os.listdir('brand_images'))
missing_images = []

for index, row in df.iterrows():
    if row['fileName'] not in image_files:
        missing_images.append((row['logoName'], row['fileName']))

# Create a set of brands to remove based on missing images
brands_to_remove = {brand for brand, _ in missing_images}

# Filter the DataFrame to exclude these brands
df_cleaned = df[~df['logoName'].isin(brands_to_remove)]

# Count occurrences of each logoName
label_counts = df_cleaned['logoName'].value_counts()
print("Label counts before filtering:", label_counts)

# Set the minimum number of samples required
min_samples = 2

# Filter out brands that do not meet the minimum sample requirement
brands_to_keep = label_counts[label_counts >= min_samples].index
df_filtered = df_cleaned[df_cleaned['logoName'].isin(brands_to_keep)]

# Print the cleaned DataFrame
print("Filtered DataFrame:", df_filtered)

# Create training and validation sets
train_df, test_df = train_test_split(
    df_filtered, test_size=0.45, random_state=42, stratify=df_filtered['logoName'])

# Update the number of unique classes based on the training data
num_classes = len(train_df['logoName'].unique())
print("Number of unique classes for model:", num_classes)

# Set parameters
img_height, img_width = 150, 150  # Adjust according to your needs
batch_size = 32

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='brand_images',
    x_col='fileName',
    y_col='logoName',
    target_size=(img_height, img_width),
    class_mode='categorical',  # Use categorical for multi-class
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='brand_images',
    x_col='fileName',
    y_col='logoName',
    target_size=(img_height, img_width),
    class_mode='categorical',  # Use categorical for multi-class
    batch_size=batch_size
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    # Match output shape to number of classes
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the model
model.save('brand_recognition_model.h5')

# Print model summary
model.summary()
