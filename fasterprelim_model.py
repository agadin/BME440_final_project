import keras
from keras import Model
from keras.api import layers
from keras.src.utils.image_utils import img_to_array, load_img
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils.numerical_utils import to_categorical
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import visualkeras

from PIL import ImageFont

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001

# Data Paths
DATASET_PATH = os.path.expanduser("~/PycharmProjects/BME440_final_project/dataset")
CATEGORIES = [
    "glioma", "glioma_tumor", "meningioma", "meningioma_tumor",
    "no_tumor", "notumor", "pituitary", "pituitary_tumor"
]

# Model Definition
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    font = ImageFont.truetype("Arial.ttf", 32)  # using comic sans is strictly prohibited!

    visualkeras.layered_view(model, to_file='img/fastprelim_model.png', legend=True, font= font).show()  # write and show
    return model

# Data Preparation
def load_data(dataset_path, subset, categories):
    images, labels = [], []
    subset_path = os.path.join(dataset_path, subset)
    for label, category in enumerate(categories):
        folder_path = os.path.join(subset_path, category)
        if not os.path.isdir(folder_path):
            print(f"Warning: Skipping missing folder: {folder_path}")
            continue
        for image_name in os.listdir(folder_path):
            try:
                image_path = os.path.join(folder_path, image_name)
                image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
                image = img_to_array(image) / 255.0
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    return np.array(images), np.array(labels)

train_images, train_labels = load_data(DATASET_PATH, "Training", CATEGORIES)
test_images, test_labels = load_data(DATASET_PATH, "Testing", CATEGORIES)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
y_val = to_categorical(y_val, num_classes=len(CATEGORIES))

# Data Augmentation
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)

# Build and compile model
model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(CATEGORIES))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Train the Model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

# Save Model
SAVE_PATH = 'brain_tumor_classifier.keras'
save_dir = os.path.dirname(SAVE_PATH) or '.'  # Default to current directory if no directory is specified
os.makedirs(save_dir, exist_ok=True)

model.save(SAVE_PATH)
print(f"Model saved successfully at {SAVE_PATH}")

# Evaluate Model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Plot Training History
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_history(history)

def plot_gradcam(image, heatmap):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap with colormap
    plt.axis("off")
    plt.title("Grad-CAM Heatmap Overlay")
    plt.show()

# Load the model and prepare for Grad-CAM
model = keras.models.load_model(SAVE_PATH)

# Grad-CAM Visualization
sample_image = x_val[0]
predicted_class = np.argmax(model.predict(np.expand_dims(sample_image, axis=0)))
print(f"Predicted Class: {CATEGORIES[predicted_class]}")

