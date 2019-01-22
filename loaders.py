import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_for_train() :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X = np.zeros((1703, 10))
    for i in range(1703):
        for j in range(10):
            X[i, j] = np.mean(X_raw[i, j])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    return (X_train, y_train, X_test, y_test)

def load_for_kaggle() :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X = np.zeros((1703, 10))
    for i in range(1703):
        for j in range(10):
            X[i, j] = np.mean(X_raw[i, j])

    X_kaggle_raw = np.load("dataset/X_test_kaggle.npy")
    X_kaggle = np.zeros((1705, 10))
    for i in range(1705):
        for j in range(10):
            X_kaggle[i , j] = np.mean(X_kaggle_raw[i, j])

    return (X, y, X_kaggle)
