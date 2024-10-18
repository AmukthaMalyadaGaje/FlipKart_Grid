import tensorflow as tf
import os

# Set parameters
data_dir = "dataset/train"  # Update this to your dataset path
img_height = 180
img_width = 180
batch_size = 32

# Function to load and preprocess images


def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    try:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_height, img_width])
        img /= 255.0  # Normalize to [0, 1]
        return img, label
    except tf.errors.InvalidArgumentError as e:
        print(f"Error processing image {path}: {e}")  # Log processing errors
        return None, label
    except Exception as e:
        print(f"Unexpected error for image {path}: {e}")
        return None, label

# Load the dataset


def load_image_dataset(data_dir):
    class_names = os.listdir(data_dir)
    print(f"Classes found: {class_names}")

    image_paths = []
    labels = []

    for label in class_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image_paths.append(img_path)
                # Use index as label for classification
                labels.append(class_names.index(label))

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_names


# Load dataset
train_ds, class_names = load_image_dataset(data_dir)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer for classes
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
try:
    history = model.fit(
        train_ds,
        epochs=10
    )
except Exception as e:
    print(f"Failed to train the model: {e}")

# Save the model
# Change this to your desired model path
model_save_path = 'classification_model.keras'
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")
