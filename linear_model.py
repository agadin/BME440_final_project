import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Set constants
IMAGE_SIZE = 150
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
                img = img.flatten() / 255.0  # Flatten image and normalize
                images.append(img)
                image_labels.append(labels.index(label))
    return np.array(images), np.array(image_labels)


def preprocess_dataset():
    train_path = os.path.join(DATASET_PATH, 'Training')
    test_path = os.path.join(DATASET_PATH, 'Testing')

    # Load training and testing datasets
    train_images, train_labels = load_images_from_directory(train_path, CLASS_NAMES, IMAGE_SIZE)
    test_images, test_labels = load_images_from_directory(test_path, CLASS_NAMES, IMAGE_SIZE)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=101
    )

    return X_train, X_val, test_images, y_train, y_val, test_labels


def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()


def plot_roc_curve(true_labels, predicted_probs, class_names):
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels == i, predicted_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def main():
    # Preprocess the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset()

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    y_test_probs = model.predict_proba(X_test)

    print("\nValidation Classification Report:\n")
    print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))

    print("\nTest Classification Report:\n")
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_test_pred, CLASS_NAMES)

    # Plot ROC curves
    plot_roc_curve(y_test, y_test_probs, CLASS_NAMES)


if __name__ == "__main__":
    main()
