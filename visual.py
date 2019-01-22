#!/usr/bin/env python

# Code ayant pour but principal de visualiser les données, sans opérations
# particulières.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_data = np.load("dataset/X_train_kaggle.npy")
training_labels = pd.read_csv("dataset/y_train_final_kaggle.csv").iloc[:,-1]

for i in range(len(training_data)) :
    print(training_labels.at[i])
    plt.plot(training_data[i][0], 'r')
    plt.show()
    plt.plot(training_data[i][1], 'g')
    plt.show()
    plt.plot(training_data[i][2], 'b')
    plt.show()
    plt.plot(training_data[i][3], 'y')
    plt.show()
