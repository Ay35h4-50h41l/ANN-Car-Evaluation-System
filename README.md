ANN Car Evaluation System

A supervised learning project that uses an Artificial Neural Network (ANN) to classify car acceptability based on structural and cost-related attributes, built on the UCI/Kaggle Car Evaluation dataset (1,728 samples).

Overview

This project implements a fully connected feedforward neural network using Keras to predict a car's evaluation class (unacceptable, acceptable, good, very good) from 6 categorical input features (buying price, maintenance cost, number of doors, capacity, luggage boot size, and safety rating).

Model Architecture
Input layer: 6 features (encoded)
Hidden layers: 4 layers, 7 neurons each, sigmoid activation
Output layer: 4-class softmax classifier
Training: 120 epochs, with class-balanced weighting to handle class imbalance in the dataset
Project Structure
├── car_evaluation.csv        # Raw dataset
├── data_preprocessing.py     # Cleans + encodes data, splits train/test
├── model.py                  # Defines the ANN architecture
├── train.py                  # Trains the model
├── evaluate.py                # Computes precision, recall, F1, accuracy
├── visualization.py          # Generates performance plots
├── app.py                    # (interface/demo — describe what this does)
├── requirements.txt          # Python dependencies
└── README.md
How It Works — Full Flow

The project runs as a pipeline. Each script depends on the previous one's output, so run them in this exact order:

Step 1: Install dependencies
bash
pip install -r requirements.txt
Step 2: Preprocess the data
bash
python data_preprocessing.py

This loads car_evaluation.csv, encodes the categorical features (buying, maint, doors, persons, lug_boot, safety) into numeric form, and splits the data into training and test sets. It typically saves the processed arrays (or returns them) for the next step to use.

Step 3: Build and train the model
bash
python train.py

This imports the ANN architecture from model.py, loads the preprocessed data, applies class-balanced weighting (since some evaluation classes have far fewer samples than others), and trains for 120 epochs. The trained model is saved (e.g. as a .h5 file) for evaluation.

Step 4: Evaluate the model
bash
python evaluate.py

Loads the trained model and test set, then computes precision, recall, F1-score, and overall accuracy — accuracy alone isn't reliable here because of class imbalance.

Step 5: Visualize results
bash
python visualization.py

Generates plots — such as class distribution, training accuracy/loss curves, and a confusion matrix — to visually inspect model performance.

Step 6 (optional): Run the app
bash
python app.py

(Explain here what app.py does — e.g. "Launches a simple interface where you can input car attributes and get a live prediction from the trained model.")

Tech Stack

Python, Keras/TensorFlow, pandas, numpy, scikit-learn, matplotlib/seaborn — developed in VS Code and Jupyter Notebook (Anaconda environment).

Dataset

UCI Car Evaluation Dataset
