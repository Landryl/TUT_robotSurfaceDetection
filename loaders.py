import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Processes the 3D input into a 2D one by computing the mean values of each signal
def generate_values(X_raw, size):
    X = np.zeros((size, 20))
    for i in range(size):
        for j in range(10):
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = X_raw[i, j].max() - X_raw[i, j].min()
    return X

def load_for_train() :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X = generate_values(X_raw, 1703)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    return (X_train, y_train, X_test, y_test, le)

def load_for_kaggle() :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = generate_values(X_raw, 1703)

    X_kaggle_raw = np.load("dataset/X_test_kaggle.npy")
    X_kaggle = generate_values(X_kaggle_raw, 1705)

    return (X, y, X_kaggle, le)
