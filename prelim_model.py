# This will be a preliminary model to ingest the images and simply predict whether the image in the dataset folder
# put all required libraries in requirements.txt
# run the following command to install all required librarie
# pip install -r requirements.txt

import kagglehub

# Download latest version and save it in the 'dataset' folder
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", path="dataset")

print("Path to dataset files:", path)
