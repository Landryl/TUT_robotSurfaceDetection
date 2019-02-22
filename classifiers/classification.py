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
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV

from utilities import loaders
from utilities import tools
from utilities import feature_extractors
import voting

XGB_installed = 1

try :
    import xgboost as xgb
    from xgboost import XGBClassifier
except :
    XGB_installed = 0


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
    print("Training KNN")
    knn = KNeighborsClassifier(5, p=2)
    knn.fit(X_train, y_train) 
    
    # Logistic Regression
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#    lr.fit(X_train, y_train) 
    
    # SVM 
    print("Training SVC")
    svm = SVC(kernel = 'linear', C = 1) #C to improve model 
    svm.fit(X_train, y_train) 
    
    # Decision Tree
    print("Training DTree")
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    
    # Random Forest
    print("Training RFC")
    rfc = RandomForestClassifier(1500)
    rfc.fit(X_train, y_train)
    
    # Naive Bayes
    print("Training GNB")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train) 
    
    # Multiclass LDA
    print("Training LDA")
    mlda = LinearDiscriminantAnalysis()
    mlda.fit(X_train, y_train)

    # Gradient Boosting
    print("Training GBC")
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)

    # Bagging
    print("Training BC")
    bc = BaggingClassifier()
    bc.fit(X_train, y_train)

    # Extra Tree
    print("Training ETC")
    etc = ExtraTreesClassifier(1000, max_features=2, max_depth=None, min_samples_split=2)
    etc.fit(X_train, y_train)

    etc_pipe = Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier(1000))),
        ('classification', ExtraTreesClassifier(1000, max_features=2, max_depth=None, min_samples_split=2))
    ])
    etc_pipe.fit(X_train, y_train)
    
    # RFECV
    print("Training RFECV")
    rfecv = RFECV(estimator = ExtraTreesClassifier(1000), verbose = 1)
    rfecv.fit(X_train, y_train)

    if(XGB_installed) :
        print("Training XGB")
        xgb_alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                            min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                            objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

        #print(" Start Feeding Data")
        #cv_folds = 5
        #early_stopping_rounds = 50
        #xgb_param = xgb_alg.get_xgb_params()
        #xgtrain = xgb.DMatrix(X_train, label=y_train)
        ## xgtest = xgb.DMatrix(X_test.values, label=y_test.values)
        #cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_alg.get_params()['n_estimators'], nfold=cv_folds,
        #                  early_stopping_rounds=early_stopping_rounds)
        #xgb_alg.set_params(n_estimators=cvresult.shape[0])
        
        #print(" Start Training")
        xgb_alg.fit(X_train, y_train, eval_metric='auc')

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

    classifiers = [('knn', knn), ('svm', svm), ('dtree', dtree), ('rfc', rfc), ('gnb', gnb), ('mlda', mlda), ('gbc', gbc), ('bc', bc), ('etc', etc), ('etc_pipe', etc_pipe), ('rfecv', rfecv)]
    if XGB_installed :
        classifiers.append(('xgb', xgb_alg))
    for classifier in classifiers :
        if classifier[0] == 'rfecv':
            nb_features = 0
            for feature in classifier[1].support_:
                if feature:
                    nb_features += 1        
            print("\nnumber of selected features: " + str(nb_features))
        
        ## Predicting the Test set results
        # Change classifier object
        print(classifier[0])
        y_pred = classifier[1].predict(X_test)
        tools.accuracy_test(y_test, y_pred)
        if classifier[0] != 'rfecv':
            tools.accuracy_average(classifier[1], X_test, y_test, 8)
        tools.conf_matrix(y_test, y_pred)
        print()

#    print("Calling the EDVC")
#    y_pred = voting.edvc([etc, rfc, gbc, xgb_alg], X_test)
#    tools.accuracy_test(y_test, y_pred)    

## Fitting and predicting for real test samples
#etc.fit(X, y)
#rfc.fit(X, y)
#gbc.fit(X, y)
rfecv.fit(X,y)
#xgb_alg.fit(X, y)
#y_kaggle = etc.predict(X_kaggle)
#y_kaggle = voting.edvc([etc, rfc, gbc, xgb_alg], X_kaggle)
y_kaggle = rfecv.predict(X_kaggle)

## Write .csv file
tools.CSVOutput(y_kaggle, le)
