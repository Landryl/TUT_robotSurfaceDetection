import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

def average_linearity(sample) :
    y = [i for i in range(len(sample))]
    A = np.vstack([sample, np.ones(len(sample))]).T
    a, b = np.linalg.lstsq(A, y, rcond=-1)[0]
    average_linearity = 0
    for i in range(len(sample)) :
        average_linearity += abs(sample[i] - a*i+b)
#    print(average_linearity/len(sample))
    return average_linearity/len(sample)


def generate_values(X_raw, size):
    X = np.zeros((size, 40))
    for i in range(size):
        for j in range(10):
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = average_linearity(X_raw[i,j])
            X[i, j+20] = X_raw[i, j].max()
            X[i, j+30] = X_raw[i, j].min()
    return X

def load_for_train(test_size, extractor=generate_values) :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = extractor(X_raw, 1703)

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    return (X_train, y_train, X_test, y_test, le)


def load_for_train_groups(test_size, extractor=generate_values):
    dataset = pd.read_csv('dataset/groups.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    groups = dataset.iloc[:, 1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X = extractor(X_raw, 1703)
    
    rs = ShuffleSplit(n_splits=9, test_size=test_size)
    #rs = GroupShuffleSplit(n_splits=4, test_size=test_size)
    #rs = StratifiedKFold(n_splits=5)

    return (rs.split(X, y, groups), le)
    

def load_for_kaggle(extractor=generate_values) :
    dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
    X_raw = np.load("dataset/X_train_kaggle.npy")
    y = dataset.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = extractor(X_raw, 1703)

    X, y = shuffle(X, y)

    X_kaggle_raw = np.load("dataset/X_test_kaggle.npy")
    X_kaggle = extractor(X_kaggle_raw, 1705)

    return (X, y, X_kaggle, le)
