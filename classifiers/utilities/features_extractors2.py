import numpy as np


X_raw = np.load("../dataset/X_train_kaggle.npy")
features_labels = [] # to remember to add them 

def features_extraction(X_raw):
    X = [[]]*len(X_raw)
    for i in range(len(X)):
        features = []
        # Mean
        for j in range(10):
            features.append(np.mean(X_raw[i, j]))
        # Variance
        for j in range(10):
            features.append(np.var(X_raw[i, j]))
        # Median
        for j in range(10):
            features.append(np.median(X_raw[i, j]))
        # Index maximum
        for j in range(10):
            features.append(np.argmax(X_raw[i, j]))
        # Index minimum
        for j in range(10):
            features.append(np.argmin(X_raw[i, j]))
        X[i] = features
    return np.array(X)



X = features_extraction(X_raw)