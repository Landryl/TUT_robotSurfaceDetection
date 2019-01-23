# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:59 2019

@author: Médéric Carriat
"""

### Classification

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import loaders
import tools
import feature_extractors


# Loading dataset
test_size = 0.25
extractor = feature_extractors.deviationer
X_train, y_train, X_test, y_test, le = loaders.load_for_train(test_size, extractor)
X, y, X_kaggle, le = loaders.load_for_kaggle(extractor)

## Creating and fitting classifier to the Training set

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5, p=2)
#knn.fit(X_train, y_train) 

# Logistic Regression
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#lr.fit(X_train, y_train) 

# SVM 
from sklearn.svm import SVC 
svm = SVC(kernel = 'rbf', C = 1)
#svm.fit(X_train, y_train) 

# Decision Tree
from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier()
#dtree.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(300)
rfc.fit(X_train, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
#gnb.fit(X_train, y_train) 

# Multiclass LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
mlda = LinearDiscriminantAnalysis()
#mlda.fit(X_train, y_train)


## Predicting the Test set results
# Change classifier object
y_pred = rfc.predict(X_test)
y_kaggle = rfc.predict(X_kaggle)

## Testing accuracy
tools.accuracy_test(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


## Write .csv file
tools.CSVOutput(y_kaggle, le)

