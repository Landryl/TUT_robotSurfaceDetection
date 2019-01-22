# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:59 2019

@author: Médéric Carriat
"""

### Classification

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

import loaders
import tools


# Loading dataset
test_size = 0.25
X_train, y_train, X_test, y_test, le = loaders.load_for_train(test_size)
X, y, X_kaggle, le = loaders.load_for_kaggle()


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
dtree.fit(X_train, y_train)

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
y_pred = dtree.predict(X_test)
y_kaggle = dtree.predict(X_kaggle)


## Testing accuracy
tools.accuracy_test(y_test, y_pred)


## Write .csv file
tools.CSVOutput(y_kaggle, le)

