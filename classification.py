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
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import loaders
import tools
import feature_extractors


dataset = pd.read_csv('dataset/groups.csv')
test_size = 0.20
i = 0

extractor = feature_extractors.euler_angles
indices_generator, le = loaders.load_for_train_groups(test_size, extractor)
X, y, X_kaggle, le = loaders.load_for_kaggle(extractor)

print_feature_importance = int(input("Afficher importance des features ? (0 ou 1) : "))

## Benchmarking classifiers over different split possibilities
for train_index, test_index in indices_generator:
    print("\nSplit {}\n".format(i))
    i += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #tools.print_train_test_repartition(dataset, train_index, test_index)

    # KNN
    knn = KNeighborsClassifier(5, p=2)
    knn.fit(X_train, y_train) 
    
    # Logistic Regression
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#    lr.fit(X_train, y_train) 
    
    # SVM 
    svm = SVC(kernel = 'linear', C = 1) #C to improve model 
    svm.fit(X_train, y_train) 
    
    # Decision Tree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    
    # Random Forest
    rfc = RandomForestClassifier(1500)
    rfc.fit(X_train, y_train)
    
    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train) 
    
    # Multiclass LDA
    mlda = LinearDiscriminantAnalysis()
    mlda.fit(X_train, y_train)

    # Gradient Boosting
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)

    # Bagging
    bc = BaggingClassifier()
    bc.fit(X_train, y_train)

    # Extra Tree
    etc = ExtraTreesClassifier(1000, max_features=2, max_depth=None, min_samples_split=2)
    etc.fit(X_train, y_train)

    if print_feature_importance :
        importances = etc.feature_importances_
        std = np.std([tree.feature_importances_ for tree in etc.estimators_],
             axis=0)
        print(importances)
        indices = [i for i in range(len(importances))]

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

    classifiers = [('knn', knn), ('svm', svm), ('dtree', dtree), ('rfc', rfc), ('gnb', gnb), ('mlda', mlda), ('gbc', gbc), ('bc', bc), ('etc', etc)]
    for classifier in classifiers :
        ## Predicting the Test set results
        # Change classifier object
        print(classifier[0])
        y_pred = classifier[1].predict(X_test)
        tools.accuracy_test(y_test, y_pred)
        tools.accuracy_average(classifier[1], X_test, y_test, 5)
        print()
     
## Fitting and predicting for real test samples
etc.fit(X, y)
y_kaggle = etc.predict(X_kaggle)

## Write .csv file
tools.CSVOutput(y_kaggle, le)
