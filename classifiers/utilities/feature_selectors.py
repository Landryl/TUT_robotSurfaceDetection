from sklearn.feature_selection import RFECV, SelectFromModel
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from utilities import tools
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from boruta import BorutaPy
from sklearn.model_selection import StratifiedKFold

### Features selector

# RFE
def rfe(X, y, X_kaggle):
    rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, class_weight=None, max_depth=7, random_state=0)
    #rf = RandomForestClassifier(100)
    #rf = ExtraTreesClassifier(200, n_jobs=-1)
    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(4), scoring='accuracy', verbose=1)
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Optimal score : {}".format(np.max(rfecv.grid_scores_)))
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    new_X = rfecv.transform(X)
    new_X_kaggle = rfecv.transform(X_kaggle)
    return new_X, new_X_kaggle

# Feature importance
def feature_imp(X, y, X_kaggle):
    clf = RandomForestClassifier(100)
    model = SelectFromModel(clf, threshold='median')
    model.fit(X, y)
    new_X = model.transform(X)
    new_X_kaggle = model.transform(X_kaggle)
    return new_X, new_X_kaggle

# PCA
def pca(X):
    pca = PCA(n_components=3)
    selection = SelectKBest(k=1)
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    new_X = combined_features.fit(X, y).transform(X)
    return new_X

# Boruta
def boruta(X, y, X_kaggle):
    rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=7)
    #rf = RandomForestClassifier(100, n_jobs=-1, class_weight=None)
    #rf = ExtraTreesClassifier(200, n_jobs=-1)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
    feat_selector.fit(X, y)
    new_X = feat_selector.transform(X)
    new_X_kaggle = feat_selector.transform(X_kaggle)
    return new_X, new_X_kaggle


# Feature union and grid search?
    
## Pipeline and GridSearch
#
#pipeline = Pipeline(
#                    [('feat_select', SelectKBest()),
#                     ('lgbm', LGBMRegressor())
#                     
#])
#
#parameters = {}
#parameters['feat_select__k'] = [5, 10]
#
#CV = GridSearchCV(pipeline, parameters, scoring = 'mean_absolute_error', n_jobs= 1)
#CV.fit(x_train_cont, y_train)   
#
#print('Best score and parameter combination = ')
#
#print(CV.best_score_)    
#print(CV.best_params_)    
#
#y_pred = CV.predict(x_valid_cont)
#print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5)))
