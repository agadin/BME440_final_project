import numpy as np
import seaborn as sns
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from keras.src.callbacks.tensorboard import TensorBoard
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.utils.numerical_utils import to_categorical
from keras.src.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D
from keras.src.layers.regularization.dropout import Dropout
from keras.src.callbacks.early_stopping import EarlyStopping
from keras.src.layers.core.dense import Dense
from keras.src.models.model import Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Set constants
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ['glioma', 'no_tumor', 'meningioma', 'pituitary']
DATASET_PATH = '/Users/colehanan/PycharmProjects/BME440_final_project/dataset'

def load_images_from_directory(data_path, labels, image_size):
    images = []
    image_labels = []
    for label in labels:
        folder_path = os.path.join(data_path, label)
        for file_name in tqdm(os.listdir(folder_path), desc=f"Loading {label} images"):
            img = cv2.imread(os.path.join(folder_path, file_name))
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                img = img / 255.0  # Normalize image to range [0, 1]
                images.append(img)
                image_labels.append(labels.index(label))
    return np.array(images), np.array(image_labels)

def preprocess_dataset():
    train_path = os.path.join(DATASET_PATH, 'Training')
    test_path = os.path.join(DATASET_PATH, 'Testing')

    # Load training and testing datasets
    train_images, train_labels = load_images_from_directory(train_path, CLASS_NAMES, IMAGE_SIZE)
    test_images, test_labels = load_images_from_directory(test_path, CLASS_NAMES, IMAGE_SIZE)

    # Shuffle and split training data into train and validation
    train_images, train_labels = shuffle(train_images, train_labels, random_state=101)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=101
    )

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=len(CLASS_NAMES))
    val_labels = to_categorical(val_labels, num_classes=len(CLASS_NAMES))
    test_labels = to_categorical(test_labels, num_classes=len(CLASS_NAMES))

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def augment_data(X_train):
    """
    Applies data augmentation to training data.
    """
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return data_gen

def build_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(rate=0.6)(x)  # Increased dropout rate to prevent overfitting
    output_layer = Dense(num_classes, activation='softmax', kernel_regularizer='l2')(x)  # Added L2 regularization
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plots a confusion matrix.
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

def main():
    # Preprocess the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset()

    # Augment the training data
    data_gen = augment_data(X_train)

    # Build the model
    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=len(CLASS_NAMES))
    model.summary()

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Callbacks for optimization
    callbacks = [
        TensorBoard(log_dir='logs'),
        ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, min_delta=0.001, mode='auto', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    # Train the model
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks
    )

    # Evaluate the model
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

    # Save test data for future use
    np.save("/Users/colehanan/PycharmProjects/BME440_final_project/test_data.npy", X_test)
    np.save("/Users/colehanan/PycharmProjects/BME440_final_project/test_labels.npy", y_test)

    # Plot the confusion matrix
    plot_confusion_matrix(true_classes, predicted_classes, CLASS_NAMES)

if __name__ == "__main__":
    main()
