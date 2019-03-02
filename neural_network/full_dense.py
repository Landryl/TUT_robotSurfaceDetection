# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:19:40 2019

@author: Sébastien Hoehn
"""

from utilities import features_extractors2, loaders, tools
import neural_networks

print("Done")

test_size = 0.20

print("Loading dataset")

extractor = features_extractors2.features_extraction
X_train, y_train, X_test, y_test, lb = loaders.load_for_train_keras(test_size, extractor)

print("Done.")

clf = neural_networks.basic(X_train[0].size, 9)
clf.fit(X_train, y_train, nb_epoch=50)

y_pred = clf.predict(X_test)
tools.accuracy_test(y_test, y_pred)