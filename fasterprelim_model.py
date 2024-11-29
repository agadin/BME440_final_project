from keras.api import layers, losses, optimizers, Sequential
from keras.src.models.model import Model
from keras.src.dtype_policies.dtype_policy import (
    DTypePolicy,
    DTypePolicy as Policy,
    dtype_policy,
    dtype_policy as global_policy,
    set_dtype_policy,
    set_dtype_policy as set_global_policy,
)
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Set global mixed precision policy
set_global_policy("mixed_float16")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATASET_PATH = os.path.expanduser("~/PycharmProjects/BME440_final_project/dataset")

# Define data augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.1),
])

def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize
    return image, label

def load_dataset(data_dir):
    class_names = sorted(
        [entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))]
    )
    file_paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                file_paths.append(file_path)
                labels.append(idx)
    return file_paths, labels, class_names

# Prepare data
train_dir = os.path.join(DATASET_PATH, 'Training')
test_dir = os.path.join(DATASET_PATH, 'Testing')

train_files, train_labels, class_names = load_dataset(train_dir)
train_labels = tf.one_hot(train_labels, depth=len(class_names))

test_files, test_labels, _ = load_dataset(test_dir)
test_labels = tf.one_hot(test_labels, depth=len(class_names))

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    .cache()
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Visualize augmented data
def visualize_data(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # Cast image to float32 for matplotlib compatibility
            image = tf.cast(images[i], tf.float32)
            plt.imshow(image.numpy())
            plt.title(class_names[tf.argmax(labels[i]).numpy()])
            plt.axis("off")
    plt.show()


visualize_data(train_dataset, class_names)

# Define CNN model
class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, (3, 3), activation="relu")
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(len(class_names), activation="softmax", dtype="float32")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=True)
        x = self.dense2(x)
        return x

model = CNNModel()

# Compile the model
model.compile(
    optimizer=LossScaleOptimizer(optimizers.Adam(learning_rate=LEARNING_RATE)),
    loss=losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_history(history)

# Save model
model.save("brain_tumor_classification_model.h5")
