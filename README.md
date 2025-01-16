# Cuneiform Translation System

This project involves the design and implementation of a system to translate ancient Persian cuneiform text into modern Persian using image processing and machine learning. The system comprises two major components: a neural network for classifying cuneiform letters and a Streamlit-based application for user interaction.

## Overview

The primary objective of this project is to leverage machine learning and computer vision techniques to create an automated tool for recognizing and translating cuneiform characters.

## Components

- 1.Training System: A neural network model trained to classify cuneiform characters.

- 2.Application: A Streamlit-based application that preprocesses input images, performs character recognition, and displays results.


## Training the Neural Network

### Dataset

Source: The dataset consists of cuneiform characters organized into training, testing, and validation sets. The dataset was initially built using template matching to identify and extract characters from scanned images of ancient cuneiform texts.

Data Augmentation: Additional data was generated by applying transformations such as rotation, scaling, and flipping to the existing samples, increasing the diversity and size of the dataset.

- Number of Classes: 37 unique cuneiform characters.

- Input Shape: (30, 30, 1) grayscale images.


### Model Architecture

The model is built using TensorFlow/Keras and consists of:

#### Convolutional Layers:

- Conv2D: Two layers with ReLU activation for feature extraction.

- MaxPooling2D: A pooling layer to reduce spatial dimensions.

- Dropout: Regularization to prevent overfitting.

- Fully Connected Layers:

#### Dense: 128 neurons with ReLU activation.

- Output layer with 37 neurons and softmax activation.

### Training Process

- Data Augmentation: Applied using ImageDataGenerator to enhance training diversity by including randomly transformed versions of the existing samples.

- Optimizer: Adam optimizer for gradient-based learning.

- Loss Function: Categorical cross-entropy.

#### Training Parameters:

- Batch Size: 128

- Epochs: 100

## Application Details

### Streamlit Application

The application allows users to upload images and view the results of the cuneiform translation.

#### Features:

- Image Upload: Accepts images in jpg, jpeg, and png formats.

- Image Preprocessing: Converts images to a format compatible with the neural network.

- Processes images using a class (CuneiformProcessor) that loads the trained model.

#### Prediction:
- Predicts cuneiform characters and maps them to modern Persian equivalents.

- visualization: Displays both the uploaded and processed images in the app interface.

#### Preprocessing:
The CuneiformProcessor class includes:

- Model Loading: Loads the trained TensorFlow model.

- Character Mapping: Maps predicted indices to Persian letters using a predefined dictionary.


## File Structure

```
project/
├── training.ipynb        # Notebook for training the neural network
├── app/                  # Application folder
│   ├── main.py           # Streamlit application script
│   ├── preprocessing.py  # Preprocessing and prediction logic
├── datasets/             # Folder containing dataset files
│   ├── train_small.zip   # Training data
│   ├── test_small.zip    # Testing data
│   ├── final_test_small  # Final testing data
├── models/               # Folder for storing trained models
│   ├── saved_model/      # Trained TensorFlow model

```

## Usage

### 1. Training the Model

- Run the training.ipynb notebook to:

- Load and preprocess the dataset.

- Train the neural network.

Save the trained model in the models/saved_model/ directory.

### 2. Running the Application

- Navigate to the app/ directory.

- Run the Streamlit application:
  ```console

  streamlit run main.py
  ```
- Upload an image and view the processed output.

### 3. Dataset Preparation

- If datasets are not included, download and place them in the datasets/ folder. Update paths in the training notebook accordingly.



## Future Work

- Improved Model Accuracy: Enhance classification performance with a larger dataset and more complex architectures.

- YOLO Integration: Use YOLO for robust letter localization.

- Real-Time Processing: Enable live recognition using a camera feed.
  
