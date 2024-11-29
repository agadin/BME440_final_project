import keras
from keras import Model
from keras.api import layers
from keras.src.utils.image_utils import img_to_array
from keras.src.utils.image_utils import load_img
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils.numerical_utils import to_categorical
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
if os.access(os.path.dirname(SAVE_PATH) or '.', os.W_OK):
    model.save(SAVE_PATH)
    print(f"Model saved successfully at {SAVE_PATH}")
else:
    print(f"Write permission denied for directory: {os.path.dirname(SAVE_PATH)}")

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


# Grad-CAM Visualization
def generate_gradcam_heatmap(model, image, class_idx):
    target_layer_name = "conv2d_2"  # Ensure this is the correct layer name
    target_layer = model.get_layer(target_layer_name)

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, 0))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def plot_gradcam(image, heatmap):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    overlay = np.uint8(0.6 * heatmap + 0.4 * image * 255)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap Overlay")
    plt.show()


# Load the model and initialize it
model = keras.models.load_model('brain_tumor_classifier.keras')
dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model.predict(dummy_input)

# Prepare sample image for Grad-CAM
sample_image = x_val[0]
sample_image = np.expand_dims(sample_image, axis=0)

# Predict class for Grad-CAM
predicted_class = np.argmax(model.predict(sample_image))
print(f"Predicted Class: {CATEGORIES[predicted_class]}")

# Generate and plot Grad-CAM
heatmap = generate_gradcam_heatmap(model, sample_image[0], predicted_class)
plot_gradcam(sample_image[0], heatmap)