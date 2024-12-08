import keras
from keras.src.models.sequential import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.dense import Dense
import collections
from keras.src.layers.core.input_layer import Input
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers.adam import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
from sklearn.metrics import classification_report, confusion_matrix

# Your existing code
DATASET_PATH = "/Users/colehanan/PycharmProjects/BME440_final_project/dataset"
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")

CATEGORY_MAPPER = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary"
}

# Renaming and consolidating folders
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

batch_size = 32
image_size = (224, 224)
learning_rate = 0.0001
epochs = 10
experiment_name = "brain_tumor_cnn"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TESTING_PATH,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_data.class_indices.keys())

# CNN Model Definition
cnn_model = Sequential([
    Input(shape=(*image_size, 3)),
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

history = cnn_model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

cnn_model.save(f"{experiment_name}.keras")  # Save in the native Keras format

# Confusion Matrix Code Added Here
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