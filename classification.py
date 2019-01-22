# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:59 2019

@author: Médéric Carriat
"""

### Classification

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle


## Functions definition

def mean_value(X_raw, size):
    X = np.zeros((size, 10))
    for i in range(size):
        for j in range(10):
            X[i, j] = np.mean(X_raw[i, j])
    return X

def accuracy_test(classifier):
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))

#Creates .cvs file
def CSVOutput(y_kaggle):
    output = le.inverse_transform(y_kaggle)
    file = open("submission.csv", "w+")
    file.write("# Id,Surface\n")
    for i in range(output.size):
        line = str(i) + "," + output[i] + "\n"
        file.write(line)
    file.close()

## Importing the dataset
dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
X_raw = np.load("dataset/X_train_kaggle.npy")
X_kaggle_raw = np.load("dataset/X_test_kaggle.npy")
y = dataset.iloc[:, -1].values


## Encoding labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


## Data processing
X = mean_value(X_raw, 1703)
shuffle(X, y)
X_kaggle = mean_value(X_kaggle_raw, 1705)


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# On split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# On full data
#X_train, y_train = X, y


## Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # object already scaled on same basis


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
svm = SVC(kernel = 'linear', C = 1)
#svm.fit(X_train, y_train) 

# Decision Tree
from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(X_train, y_train) 
  

## Predicting the Test set results
# Change classifier object
y_pred = dtree.predict(X_test)
y_kaggle = dtree.predict(X_kaggle)


## Testing accuracy
accuracy_test(dtree)


## Write .csv file
CSVOutput(y_kaggle)

