# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam

# # Define paths
# train_dir = 'C:\\Users\\devad\\Downloads\\archive (1).zip\\dataset\\dataset\\train'
# test_dir = 'C:\\Users\\devad\\Downloads\\archive (1).zip\\dataset\\dataset\\test'

# # Image preprocessing
# img_width, img_height = 150, 150
# batch_size = 32

# train_datagen = ImageDataGenerator(rescale=1.0/255)
# test_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )

# # Build the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu',
#            input_shape=(img_width, img_height, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')  # Binary classification
# ])

# # Compile the model
# model.compile(optimizer=Adam(), loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, epochs=10, validation_data=test_generator)

# # Save the model
# model.save('data/saved_model/shelf_life_model.h5')
# print("Model trained and saved successfully.")
