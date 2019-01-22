# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:59 2019

@author: Médéric Carriat
"""

### Classification

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle

import loaders

## Functions definition

#Prints accuracy values of models 
def accuracy_test(classifier):
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))

#Creates .cvs output file
def CSVOutput(y_kaggle):
    output = le.inverse_transform(y_kaggle)
    file = open("submission.csv", "w+")
    file.write("# Id,Surface\n")
    for i in range(output.size):
        line = str(i) + "," + output[i] + "\n"
        file.write(line)
    file.close()


# Loading dataset

X_train, y_train, X_test, y_test, le = loaders.load_for_train()
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
accuracy_test(dtree)


## Write .csv file
CSVOutput(y_kaggle)

