from keras.src.models.sequential import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers.adam import Adam
from keras.src.layers.core.input_layer import Input
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import collections
import cv2
from keras.src.layers.normalization.batch_normalization import (
    BatchNormalization,
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATASET_PATH = "/Users/colehanan/PycharmProjects/BME440_final_project/dataset"
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")


# Define preprocessing function
def preprocess_image(img):
    # Resize image
    img = cv2.resize(img, (150, 150))

    # Check if the image is already grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to grayscale only if it's a 3-channel image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Ensure the image is of type uint8
    if img.dtype != 'uint8':
        img = (img * 255).astype('uint8')

    # Apply Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply contrast enhancement
    img = cv2.equalizeHist(img)

    # Normalize pixel values
    img = img / 255.0

    # Expand dimensions to (150, 150, 1)
    img = np.expand_dims(img, axis=-1)

    return img

# Custom ImageDataGenerator
class PreprocessedImageDataGenerator(ImageDataGenerator):
    def __init__(self, preprocessing_function=None, **kwargs):
        super().__init__(preprocessing_function=preprocessing_function, **kwargs)

    def flow_from_directory(self, directory, **kwargs):
        return PreprocessedDirectoryIterator(directory, self, **kwargs)


class PreprocessedDirectoryIterator(tf.keras.preprocessing.image.DirectoryIterator):
    def __init__(self, directory, image_data_generator, **kwargs):
        super().__init__(directory, image_data_generator, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
        batch_x = np.array([self.image_data_generator.preprocessing_function(img) for img in batch_x])
        return batch_x, batch_y


# Set up data generators
batch_size = 32
img_size = (150, 150)
learning_rate = 0.0001
epochs = 50
experiment_name = "brain_tumor_cnn_preprocessed"

train_datagen = PreprocessedImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

validation_generator = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

test_datagen = PreprocessedImageDataGenerator(preprocessing_function=preprocess_image)

test_generator = test_datagen.flow_from_directory(
    TESTING_PATH,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

class_labels = list(train_generator.class_indices.keys())

# Build the CNN model
model = Sequential([
    Input(shape=(*img_size, 1)),  # Changed to 1 channel for grayscale
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set up callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=f"{experiment_name}_best.keras",
    save_best_only=True,
    monitor='val_loss'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping]
)

# Save the model
model.save(f"{experiment_name}.keras")

# ... (rest of the code remains the same)