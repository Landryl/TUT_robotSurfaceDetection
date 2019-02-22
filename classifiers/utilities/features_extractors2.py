import numpy as np
from scipy.stats import skew, kurtosis

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
        # Standard deviation
        for j in range(10):
            features.append(np.std(X_raw[i, j]))
        # Minimum
        for j in range(10):
            features.append(np.min(X_raw[i, j]))
        # Maximum
        for j in range(10):
            features.append(np.max(X_raw[i, j]))
        # Skewness
        for j in range(10):
            features.append(skew(X_raw[i, j]))
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
        # Kurtosis
        for j in range(10):
            features.append(kurtosis(X_raw[i, j]))
    return np.array(X)



X = features_extraction(X_raw)