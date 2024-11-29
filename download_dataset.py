import os
import zipfile
import subprocess

# Ensure the dataset directory exists
os.makedirs('dataset', exist_ok=True)

# Path to save the dataset zip
zip_path = os.path.join('dataset', 'brain-tumor-mri-dataset.zip')

# Use the Kaggle CLI to download the dataset
try:
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'masoudnickparvar/brain-tumor-mri-dataset', '-p', 'dataset'],
        check=True
    )
except FileNotFoundError:
    raise RuntimeError("The Kaggle CLI is not installed or not found in your PATH. Install it using `pip install kaggle`.")

# Unzip the downloaded file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Remove the zip file after extraction
os.remove(zip_path)

print(f"Dataset downloaded, unzipped, and saved to 'dataset' folder")
