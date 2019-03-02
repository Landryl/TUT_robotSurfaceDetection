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
clf.fit(X_train, y_train, nb_epoch=50, validation_data=(X_test, y_test))

y_pred = clf.predict(X_test)

print("▶ Evaluating ◀")
y_pred = lb.inverse_transform(tools.max_one_hot(y_pred), 0.5)
y_test = lb.inverse_transform(y_test, 0.5)
tools.accuracy_test(y_test, y_pred)
