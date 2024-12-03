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
from PIL import ImageFont
import visualkeras

def get_prefix(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(filename)[0])

def rename_functions(filename, prefix):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    pattern = r'\bdef\s+(\w+)'
    return re.sub(pattern, lambda m: f'def {prefix}_{m.group(1)}', content)

def handle_imports(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    pattern = r'^(from|import)\s+.*$'
    return re.sub(pattern, lambda m: handle_import_statement(m.group(0), filename), content, flags=re.MULTILINE)

def handle_import_statement(statement, filename):
    match = re.match(r'^(from|import)\s+(\w+)', statement)
    if match:
        module = match.group(2)
        if f"{module}.py" in files:
            prefix = get_prefix(f"{module}.py")
            return statement.replace(module, prefix)
    return statement

def merge_files(files, output):
    with open(output, "w", encoding="utf-8") as outfile:
        for filename in files:
            prefix = get_prefix(filename)
            content = rename_functions(filename, prefix)
            content = handle_imports(filename)
            outfile.write(f"# Merging {filename}\n")
            outfile.write(content)
            outfile.write("\n")

# Your existing code
DATASET_PATH = "./dataset"
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
epochs = 5
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
font = ImageFont.truetype("Arial.ttf", 32)
visualkeras.layered_view(cnn_model, to_file='img/midmodelv3.png', legend=True,
                         font=font).show()

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

font = ImageFont.truetype("Arial.ttf", 32)
visualkeras.layered_view(cnn_model, to_file='img/midmodelv3.png', legend=True,
                         font=font).show()


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

results_df = pd.DataFrame({
    "Experiment": [experiment_name],
    "Test Accuracy": [test_accuracy],
    "Test Loss": [test_loss],
    "Epochs": [epochs],
    "Learning Rate": [learning_rate],
    "Batch Size": [batch_size]
})

results_csv = f"{experiment_name}_results.csv"
if os.path.exists(results_csv):
    previous_results = pd.read_csv(results_csv)
    results_df = pd.concat([previous_results, results_df], ignore_index=True)

results_df.to_csv(results_csv, index=False)

print("Results saved to:", results_csv)

class_weights = {i: len(train_data.classes) / (len(np.unique(train_data.classes)) * np.sum(train_data.classes == i))
                 for i in range(len(np.unique(train_data.classes)))}

print(collections.Counter(train_data.classes))
print(collections.Counter(test_data.classes))

# Merge the files
files = ["file1.py", "file2.py"]  # Replace with your actual file names
output = "merged_script.py"
merge_files(files, output)

print(f"Files merged into {output}")

# Execute the merged file
try:
    exec(open(output, encoding="utf-8").read())
    print(f"The merged file {output} is runnable.")
except Exception as e:
    print(f"The merged file {output} is not runnable.")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(y_test_new, pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
                cmap=colors_green[::-1], alpha=0.7, linewidths=2, linecolor=colors_dark[3])
    fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
             fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

    plt.show()