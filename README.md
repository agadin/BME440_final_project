# Overview
This repository will explore the use of machine learning models to predict tumor type using brain scan images.

# Data
The data used in this project is from the [Brain Tumor Classification dataset](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) on Kaggle. This dataset contains 3,253 brain MRI images of 4 classes: glioma, meningioma, pituitary, and no tumor. The images are in .jpg format and are 512x512 pixels.

## Downloading the Data
To download the data, you can use the Kaggle API. First, you need to install kagglehub by running the following command:
```
pip install kaggle
```
Then, you need to create a Kaggle API key by going to your Kaggle account, clicking on "Create New API Token", and saving the file to your computer in the folder at `~/.kaggle` (to find this folder easily on mac use the press command+shift+g and input `~/.kaggle`). Finally, you can download the data by running the `dowbload_dataset.py` script in pycharm.


## timeline
1. Develop prelim model to predict tumor type using brain scan images 
2. Develop new pre-processing techniques to improve prelim model performance 
3. Develop new model to predict tumor type using brain scan images using the new pre-processing techniques. This model will be compared to the prelim model. This model will also output visual characteristics that are influential for the model to choose one output over another. 
4. Develop a model that is not a neural network to predict tumor type using brain scan images. This model will be compared to the prelim model and the model developed using the new pre-processing techniques. This model will also output visual characteristics that are influential for the model to choose one output over another. 
5. Devlop a unique visualisation of the model design and steps for presentation. 
6. Develop a presentation that compares and contrasts the models developed in this project.


## Unique approaches
* Output specific visual characteristics that are influential for the model to choose one output over another.
* Unique visualisation of the model design and steps for presentation.
* Have multiple models to compare and contrast.
* Use of a model that is not a neural network.

## BME440 Specific Elements
* Model presented in class