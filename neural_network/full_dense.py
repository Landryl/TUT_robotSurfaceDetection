# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:19:40 2019

@author: Sébastien Hoehn
"""

print("▶ Importing librairies...")
from utilities import features_extractors2, loaders, tools, feature_selectors
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import neural_networks    


test_size = 0.20

print("▶ Loading dataset ◀")

X_raw = np.load("./dataset/X_train_kaggle.npy")
X_kaggle_raw = np.load("./dataset/X_test_kaggle.npy")
dataset = pd.read_csv("./dataset/groups.csv")
y = dataset.iloc[:, -1].values
groups = dataset.iloc[:, 1].values

#ohe = OneHotEncoder(sparse=False)
#y = y.reshape(-1, 1)
#y = ohe.fit_transform(y)

y_boruta = y

lb = LabelBinarizer()
y = y.reshape(-1, 1)
y = lb.fit_transform(y)

bor = LabelEncoder()
y_boruta = bor.fit_transform(y_boruta)


print("▶ Extracting and selecting features ◀")
# Without orientation 
#X = features_extractors2.features_extraction_no_ori(X_raw, 42)
#X_kaggle = features_extractors2.features_extraction_no_ori(X_kaggle_raw, 42)

# With orientation
X = features_extractors2.features_extraction(X_raw, 42)
X_kaggle = features_extractors2.features_extraction(X_kaggle_raw, 42)

## Feature selection
#X, X_kaggle = feature_selectors.rfe(X, y_boruta, X_kaggle)
#X = feature_selectors.pca(X)
X, X_kaggle = feature_selectors.boruta(X, y_boruta, X_kaggle)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=3)


print("▶ Training neural network ◀")
clf = neural_networks.basic(X_train[0].size, 9)
history = clf.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))


print("▶ Evaluating ◀")
y_pred = clf.predict(X_test)
y_pred = lb.inverse_transform(tools.max_one_hot(y_pred), 0.5)
y_test = lb.inverse_transform(y_test, 0.5)
tools.plot_history(history)
tools.accuracy_test(y_test, y_pred)
tools.conf_matrix(y_test, y_pred)