import numpy as np
import seaborn as sns
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
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

# Set constants
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10
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


def build_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model


def plot_training_history(history):
    """
    Plots training and validation accuracy/loss over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_roc_curve(true_labels, predicted_probs, class_names):
    """
    Plots ROC curves for each class.
    """
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predicted_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def create_best_epoch_table(history):
    """
    Creates a table displaying the stats from the best epoch.
    """
    best_epoch_idx = np.argmax(history.history['val_accuracy'])
    best_epoch_data = {
        "Metric": ["Accuracy", "Loss", "Val Accuracy", "Val Loss"],
        "Value": [
            history.history['accuracy'][best_epoch_idx],
            history.history['loss'][best_epoch_idx],
            history.history['val_accuracy'][best_epoch_idx],
            history.history['val_loss'][best_epoch_idx]
        ]
    }
    best_epoch_df = pd.DataFrame(best_epoch_data)
    print("\nBest Epoch Stats Table:")
    print(best_epoch_df)

    return best_epoch_df


def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plots a confusion matrix.
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

def main():
    # Preprocess the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset()

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
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
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

    # Plot training history
    plot_training_history(history)

    # Create a table for the best epoch
    create_best_epoch_table(history)

    # Plot ROC curves
    plot_roc_curve(y_test, predictions, CLASS_NAMES)

if __name__ == "__main__":
    main()