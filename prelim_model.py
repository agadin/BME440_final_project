import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATASET_PATH = os.path.expanduser("~/PycharmProjects/BME440_final_project/dataset")  # Adapted path to your dataset

# Define data augmentation using tf.image
def preprocess_image(file_path, label, augment=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
    image /= 255.0  # Normalize
    return image, label

# Load dataset from directory
def load_dataset(data_dir):
    class_names = sorted(
        [
            entry for entry in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, entry))
        ]
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

train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y, augment=True)).batch(BATCH_SIZE).shuffle(1000)

test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_dataset = test_dataset.map(lambda x, y: preprocess_image(x, y)).batch(BATCH_SIZE)

# Define model
class CNNModel(tf.Module):
    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(len(class_names), activation="softmax")

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

model = CNNModel()

# Loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training step
@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Evaluation step
@tf.function
def evaluate_model(model, dataset):
    correct, total = 0, 0
    for images, labels in dataset:
        predictions = model(images, training=False)
        correct += tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1) == tf.argmax(labels, axis=1), tf.float32))
        total += images.shape[0]
    return correct / total

# Training loop
for epoch in range(EPOCHS):
    for images, labels in train_dataset:
        train_loss = train_step(model, images, labels)

    val_acc = evaluate_model(model, test_dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss.numpy():.4f}, Val Accuracy: {val_acc.numpy():.4f}")

# Save model
tf.saved_model.save(model, "brain_tumor_classification_model")
