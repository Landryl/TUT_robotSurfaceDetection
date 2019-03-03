"""
Testing McFly model choosing
"""

import sys
import os
import numpy as np
import pandas as pd
import json
# mcfly
from mcfly import modelgen, find_architecture, storage
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from keras.models import load_model
from utilities import feature_extractors, tools
result_path = "./results_mcfly/"

print("▶ Loading dataset ◀")
X_raw = np.load("./dataset/X_train_kaggle.npy")
X_kaggle_raw = np.load("./dataset/X_test_kaggle.npy")
dataset = pd.read_csv("./dataset/groups.csv")
y = dataset.iloc[:, -1].values
groups = dataset.iloc[:, 1].values

lb = LabelBinarizer()
y = y.reshape(-1, 1)
y = lb.fit_transform(y)

#X = feature_extractors.identity(X_raw, 42)
#X_kaggle = feature_extractors.identity(X_kaggle_raw, 42)

X = feature_extractors.RNN_extractor(X_raw, 42)
X_kaggle = feature_extractors.RNN_extractor(X_kaggle_raw, 42)


print("▶ Splitting data set ◀")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    
print("▶ Generating model ◀")    
models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=9,
                                  number_of_models=40)


print("▶ Finding best architecture and saving it ◀")
if not os.path.exists(result_path):
        os.makedirs(result_path)
        
outputfile = os.path.join(result_path, 'modelcomparison.json')
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train,
                                                                           X_val, y_val,
                                                                           models,nr_epochs=5,
                                                                           subset_size=350,
                                                                           verbose=True,
                                                                           batch_size=32,
                                                                           outputfile=outputfile,
                                                                           early_stopping=True)

print('Details of the training process were stored in ', outputfile)

best_model_index = np.argmax(val_accuracies)
best_model, best_params, best_model_types = models[best_model_index]
print('Model type and parameters of the best model:')
print(best_model_types)
print(best_params)


print("▶ Training best model ◀")
nr_epochs = 50
history = best_model.fit(X_train, y_train, epochs=nr_epochs, validation_data=(X_val, y_val))

print("▶ Saving best model ◀")
best_model.save(os.path.join(result_path, 'best_model.h5'))

#best_model = load_model(os.path.join(result_path, 'best_model.h5'))

print("▶ Evaluating ◀")
score_test = best_model.evaluate(X_test, y_test, verbose=True)
print('Score of best model: ' + str(score_test))

y_pred = lb.inverse_transform(tools.max_one_hot(best_model.predict(X_test)), 0.5)
y_test = lb.inverse_transform(y_test, 0.5)
tools.accuracy_test(y_test, y_pred)
tools.conf_matrix(y_test, y_pred)
tools.plot_history(history)