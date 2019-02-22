from sklearn.feature_selection import RFECV
from utilities import features_extractors2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Load data
dataset = pd.read_csv('dataset/y_train_final_kaggle.csv')
X_raw = np.load("dataset/X_train_kaggle.npy")
y = dataset.iloc[:, -1].values

# Encode y
le = LabelEncoder()
y = le.fit_transform(y)

# Extract features
X = features_extractors2.features_extraction_no_ori(X_raw)

# Declare classifiers
etc = ExtraTreesClassifier(200)
lr = LogisticRegression()
mlda = LinearDiscriminantAnalysis()

# Feature elimination
rfecv = RFECV(estimator=etc, step=1, cv=StratifiedKFold(4), scoring='accuracy', verbose=1)
rfecv.fit(X, y)

# Results
print("Optimal number of features : %d" % rfecv.n_features_)
print("Optimal score : {}".format(np.max(rfecv.grid_scores_)))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
