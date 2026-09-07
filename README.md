# ANN-Car-Evaluation-System
A supervised learning project that uses an Artificial Neural Network (ANN) to classify car acceptability based on structural and cost-related attributes, built on the UCI/Kaggle Car Evaluation dataset (1,728 samples).

Overview
This project implements a fully connected feedforward neural network using Keras to predict a car's evaluation class (unacceptable, acceptable, good, very good) from 6 categorical input features (buying price, maintenance cost, number of doors, capacity, luggage boot size, and safety rating).

Model Architecture
Input layer: 6 features (encoded)
Hidden layers: 4 layers, 7 neurons each, sigmoid activation
Output layer: 4-class softmax classifier
Training: 120 epochs, with class-balanced weighting to handle class imbalance in the dataset
Pipeline

The project follows a modular structure for clarity and reusability:

File	Purpose
data_preprocessing.py	Loads and encodes the raw dataset, handles train/test split
model.py	Defines the ANN architecture
train.py	Trains the model with class weighting
evaluate.py	Computes precision, recall, F1-score, and accuracy
visualize.py	Generates plots for model performance and class distribution
Evaluation

Model performance is assessed using precision, recall, F1-score, and overall accuracy — chosen over accuracy alone due to class imbalance in the target labels.

Tech Stack

Python, Keras/TensorFlow, pandas, numpy, scikit-learn, matplotlib/seaborn — developed in VS Code and Jupyter Notebook (Anaconda environment).

Dataset

UCI Car Evaluation Dataset
