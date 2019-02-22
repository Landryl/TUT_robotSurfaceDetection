import numpy as np
from scipy.stats import skew, iqr, kurtosis
from statsmodels.robust import mad
from sklearn.preprocessing import MinMaxScaler

features_labels = [] # to remember to add them 

def normalize(X_raw):
    X = []
    scaler = MinMaxScaler()
    for row in X_raw:
       scaler.fit(row)
       X.append(scaler.transform(row))
    return np.array(X)
        
def sma(x, y, z):
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return sum/len(x)

def features_labels():
    fl = []
    channels = ["O-x", "O-y", "O-z", "O-w", "AV-x", "AV-y", "AV-z", "LA-x", "LA-y", "LA-z"]
    labels = ['mean', 'var', 'std', 'min', 'max', 'skew', 'median', 'i_max', 'imin', 'inter_range', 'mad', 'kurto', 'rms']
    for channel in channels:
        for label in labels:
            fl.append("{} {}".format(channel, label))
    fl.append('AV sma')
    fl.append('LA sma')
    return fl

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
        # Interquartile range
        for j in range(10):
            features.append(iqr(X_raw[i, j]))
        # Mean absolute deviation
        for j in range(10):
            features.append(mad(X_raw[i, j]))
        # Kurtosis
        for j in range(10):
            features.append(kurtosis(X_raw[i, j]))
        # RMS
        for j in range(10):
            features.append(np.sqrt(np.mean(np.square(X_raw[i, j]))))      
        # Signal magnitude area
        features.append(sma(X_raw[i, 4], X_raw[i, 5], X_raw[i, 6]))
        features.append(sma(X_raw[i, 7], X_raw[i, 8], X_raw[i, 9]))
        
        # Other extractors usable if we fix them
        # Entropy
        #X_normalized = normalize(X_raw)
        #for j in range(10):
        #    features.append(entropy(X_normalized[i, j]))
        
        X[i] = features
    
    return np.array(X)

def features_extraction_no_ori(X_raw):
    X = [[]]*len(X_raw)
    for i in range(len(X)):
        features = []
        # Mean
        for j in range(4, 10):
            features.append(np.mean(X_raw[i, j]))
        # Variance
        for j in range(4, 10):
            features.append(np.var(X_raw[i, j]))
        # Standard deviation
        for j in range(4, 10):
            features.append(np.std(X_raw[i, j]))
        # Minimum
        for j in range(4, 10):
            features.append(np.min(X_raw[i, j]))
        # Maximum
        for j in range(4, 10):
            features.append(np.max(X_raw[i, j]))
        # Skewness
        for j in range(4, 10):
            features.append(skew(X_raw[i, j]))
        # Median
        for j in range(4, 10):
            features.append(np.median(X_raw[i, j]))
        # Index maximum
        for j in range(4, 10):
            features.append(np.argmax(X_raw[i, j]))
        # Index minimum
        for j in range(4, 10):
            features.append(np.argmin(X_raw[i, j]))
        # Interquartile range
        for j in range(4, 10):
            features.append(iqr(X_raw[i, j]))
        # Mean absolute deviation
        for j in range(4, 10):
            features.append(mad(X_raw[i, j]))
        # Kurtosis
        for j in range(4, 10):
            features.append(kurtosis(X_raw[i, j]))
        # RMS
        for j in range(4, 10):
            features.append(np.sqrt(np.mean(np.square(X_raw[i, j]))))      
        # Signal magnitude area
        features.append(sma(X_raw[i, 4], X_raw[i, 5], X_raw[i, 6]))
        features.append(sma(X_raw[i, 7], X_raw[i, 8], X_raw[i, 9]))
        
        # Other extractors usable if we fix them
        # Entropy
        #X_normalized = normalize(X_raw)
        #for j in range(10):
        #    features.append(entropy(X_normalized[i, j]))
        
        X[i] = features
    
    return np.array(X)


