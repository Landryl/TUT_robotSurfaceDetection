# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:19:40 2019

@author: Sébastien Hoehn
"""

from utilities import features_extractors2, loaders, tools, feature_selectors
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import neural_networks

def load_for_train_keras(test_size) :
    ## Load data
    X_raw = np.load("./dataset/X_train_kaggle.npy")
    X_kaggle_raw = np.load("./dataset/X_test_kaggle.npy")
    dataset = pd.read_csv("./dataset/groups.csv")
    y = dataset.iloc[:, -1].values
    groups = dataset.iloc[:, 1].values

    #ohe = OneHotEncoder(sparse=False)
    #y = y.reshape(-1, 1)
    #y = ohe.fit_transform(y)

    lb = LabelBinarizer()
    y = y.reshape(-1, 1)
    y = lb.fit_transform(y)
    
    ## Extract features
    # Without orientation 
    #X = features_extractors2.features_extraction_no_ori(X_raw)
    #X_kaggle = features_extractors2.features_extraction_no_ori(X_kaggle_raw)
    
    # With orientation
    X = features_extractors2.features_extraction(X_raw, 1703)
    X_kaggle = features_extractors2.features_extraction(X_kaggle_raw, 1703)

    ## Feature selection
    #, X_kaggle = feature_selectors.rfe(X, y, X_kaggle)
    #X = feature_selectors.pca(X)
    X, X_kaggle = feature_selectors.boruta(X, y, X_kaggle)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    return (X_train, y_train, X_test, y_test, lb)

print("Done")

test_size = 0.20

print("Loading dataset")

extractor = features_extractors2.features_extraction
X_train, y_train, X_test, y_test, lb = load_for_train_keras(test_size, extractor)

print("Done.")

clf = neural_networks.basic(X_train[0].size, 9)
clf.fit(X_train, y_train, nb_epoch=50, validation_data=(X_test, y_test))

y_pred = clf.predict(X_test)

print("▶ Evaluating ◀")
y_pred = lb.inverse_transform(tools.max_one_hot(y_pred), 0.5)
y_test = lb.inverse_transform(y_test, 0.5)
tools.accuracy_test(y_test, y_pred)
