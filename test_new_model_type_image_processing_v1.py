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
from keras.models import load_model

# Set constants
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 1
CLASS_NAMES = ['glioma', 'no_tumor', 'meningioma', 'pituitary']
DATASET_PATH = './dataset'


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
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()


def occlusion_sensitivity(model, image, occlusion_size=15, stride=10):
    heatmap = np.zeros((image.shape[0], image.shape[1]))
    original_pred = model.predict(np.expand_dims(image, axis=0))[0]
    class_idx = np.argmax(original_pred)

    for y in range(0, image.shape[0] - occlusion_size + 1, stride):
        for x in range(0, image.shape[1] - occlusion_size + 1, stride):
            occluded_image = image.copy()
            occluded_image[y:y+occlusion_size, x:x+occlusion_size, :] = 0
            pred = model.predict(np.expand_dims(occluded_image, axis=0))[0]
            heatmap[y:y+occlusion_size, x:x+occlusion_size] = original_pred[class_idx] - pred[class_idx]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    return heatmap





def save_occlusion_video_across_epochs(images, output_dir, epochs, model_dir, occlusion_size=15, stride=10):
    os.makedirs(output_dir, exist_ok=True)

    for img_idx, image in enumerate(images):
        # Video writer for this image
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(output_dir, f"occlusion_image_{img_idx + 1}.avi")
        video = cv2.VideoWriter(video_path, fourcc, 1, (image.shape[1], image.shape[0]))

        print(f"Processing image {img_idx + 1}/{len(images)}...")

        for epoch in range(1, epochs + 1):
            print(f" - Epoch {epoch}")

            # Load the model for the current epoch
            model_path = os.path.join(model_dir, f"model_epoch_{epoch:02d}.h5")
            model = load_model(model_path)

            # Generate occlusion sensitivity heatmap
            heatmap = occlusion_sensitivity(model, image, occlusion_size, stride)
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            combined_frame = cv2.addWeighted((image * 255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)

            # Annotate frame with epoch number
            cv2.putText(
                combined_frame,
                f"Epoch {epoch}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            video.write(combined_frame)

        video.release()
        print(f"Video saved at {video_path}")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset()

    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=len(CLASS_NAMES))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir='logs'),
        ModelCheckpoint(os.path.join(model_dir, "model_epoch_{epoch:02d}.h5"), save_best_only=False),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=BATCH_SIZE, callbacks=callbacks)

    # Generate videos for six selected images from the test set
    selected_images = X_test[:6]
    save_occlusion_video_across_epochs(selected_images, "./occlusion_videos", epochs=12, model_dir=model_dir)


if __name__ == "__main__":
    main()
