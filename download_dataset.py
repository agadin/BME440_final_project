import os
import requests
import json
import zipfile

# Ensure the dataset directory exists
os.makedirs('dataset', exist_ok=True)

# Load Kaggle API credentials
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    kaggle_api = json.load(f)

# Define the URL and headers for the request
url = 'https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset'
headers = {
    'Authorization': f"Bearer {kaggle_api['key']}"
}

# Download the dataset
response = requests.get(url, headers=headers, stream=True)
zip_path = os.path.join('dataset', 'brain-tumor-mri-dataset.zip')

# Save the dataset to the specified path
with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Unzip the downloaded file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('dataset')

# Remove the zip file after extraction
os.remove(zip_path)

print(f"Dataset downloaded, unzipped, and saved to 'dataset' folder")