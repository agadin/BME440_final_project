import warnings
import re
import cv2
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import DirectoryIterator
from keras.src.optimizers.adam import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.models.sequential import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.dense import Dense
import collections
from keras.src.layers.core.input_layer import Input
from keras.src.layers.regularization.dropout import Dropout
from keras.src.callbacks.early_stopping import EarlyStopping

warnings.filterwarnings("ignore")

# Define preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != 'uint8':
        img = (img * 255).astype('uint8')
    img = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

class PreprocessedImageDataGenerator(ImageDataGenerator):
    def __init__(self, preprocessing_function=None, **kwargs):
        super().__init__(preprocessing_function=preprocessing_function, **kwargs)

    def flow_from_directory(self, directory, **kwargs):
        return PreprocessedDirectoryIterator(directory, self, **kwargs)

class PreprocessedDirectoryIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator, **kwargs):
        super().__init__(directory, image_data_generator, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
        batch_x = np.array([self.image_data_generator.preprocessing_function(img) for img in batch_x])
        return batch_x, batch_y

DATASET_PATH = "/Users/colehanan/PycharmProjects/BME440_final_project/dataset"
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")

CATEGORY_MAPPER = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "no_tumor",
    "pituitary": "pituitary",
}

for path in [TRAINING_PATH, TESTING_PATH]:
    for folder in os.listdir(path):
        original_path = os.path.join(path, folder)
        consolidated_name = CATEGORY_MAPPER.get(folder, folder)
        consolidated_path = os.path.join(path, consolidated_name)
        if original_path != consolidated_path and os.path.isdir(original_path):
            if not os.path.exists(consolidated_path):
                os.rename(original_path, consolidated_path)
            else:
                for file in os.listdir(original_path):
                    os.rename(
                        os.path.join(original_path, file),
                        os.path.join(consolidated_path, file)
                    )
                os.rmdir(original_path)

batch_size = 64
image_size = (224, 224)
learning_rate = 0.001
epochs = 10
experiment_name = "brain_tumor_cnn"

# Use the custom PreprocessedImageDataGenerator
train_datagen = PreprocessedImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = PreprocessedImageDataGenerator(preprocessing_function=preprocess_image)

train_data = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TESTING_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

class_labels = list(test_data.class_indices.keys())

cnn_model = Sequential([
    Input(shape=(*image_size, 1)),
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_labels), activation='sigmoid')
])

cnn_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

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

history = cnn_model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

cnn_model.save(f"{experiment_name}.keras")

# Plotting accuracy and loss
def plot_training_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_metrics(history)

test_loss, test_accuracy = cnn_model.evaluate(test_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = cnn_model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=1))