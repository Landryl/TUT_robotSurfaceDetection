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

from utilities import loaders
from utilities import tools
from utilities import feature_extractors


dataset = pd.read_csv('dataset/groups.csv')
test_size = 0.20
i = 0

extractors = feature_extractors.get_all_extractors()

for extractor in extractors :
    print(extractor[0])
    indices_generator, le = loaders.load_for_train_groups(test_size, extractor[1])
    X, y, X_kaggle, le = loaders.load_for_kaggle(extractor[1])

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
        lr.fit(X_train, y_train) 
    
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
        etc = ExtraTreesClassifier(500)
        etc.fit(X_train, y_train)
    
        classifiers = [('knn', knn), ('lr', lr), ('svm', svm), ('dtree', dtree), ('rfc', rfc), ('gnb', gnb), ('mlda', mlda), ('gbc', gbc), ('bc', bc), ('etc', etc)]
        for classifier in classifiers :
            ## Predicting the Test set results
            # Change classifier object
            print(classifier[0])
            y_pred = classifier[1].predict(X_test)
            tools.accuracy_test(y_test, y_pred)
            print()
        break
