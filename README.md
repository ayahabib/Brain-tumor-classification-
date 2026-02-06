## Brain Tumor Classification using Deep Learning

This project focuses on automatic brain tumor classification from MRI images using Convolutional Neural Networks (CNNs). The goal is to assist in early detection and diagnosis by accurately classifying brain MRI scans into tumor categories.

## Problem Statement

Manual analysis of brain MRI images is time-consuming and highly dependent on expert radiologists. This project leverages Computer Vision and Deep Learning techniques to automatically classify brain tumors, improving efficiency and supporting medical decision-making.

## Methodology

Used MRI brain images dataset for training and evaluation

Applied image preprocessing and data augmentation to enhance model generalization

Built and trained a CNN-based classification model using TensorFlow & Keras

Optimized the model using techniques such as dropout and batch normalization

Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrix

## Results

Achieved high classification accuracy on validation data

Demonstrated strong performance across multiple tumor classes

Visualized predictions and model confidence for better interpretability

## Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib, Seaborn

Streamlit

## Key Features

End-to-end pipeline: preprocessing → training → evaluation

Multi-class brain tumor classification

Clean and modular code structure

Easily extendable to other medical imaging tasks

## Dataset

https://www.kaggle.com/code/pkdarabi/brain-tumor-detection-by-cnn-pytorch/input
https://www.kaggle.com/datasets/umarsiddiqui9/bttypes

## Trained Model
The trained model file is not included in this repository due to size limitations.
You can train the model by running the provided notebook/script.

Future work may include hosting the trained model using cloud storage.


## Future Improvements

Experiment with transfer learning (ResNet, VGG, EfficientNet)

Improve performance using hyperparameter tuning

Deploy the model as a web or desktop application

Extend to tumor segmentation tasks
