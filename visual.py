#!/usr/bin/env python

# Code ayant pour but principal de visualiser les données, sans opérations
# particulières.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_data = np.load("dataset/X_train_kaggle.npy")
training_labels = pd.read_csv("dataset/y_train_final_kaggle.csv").iloc[:,-1]

#L = [0, ]
#
#for k in L:
#    for i in range(10) :
#        print(training_labels.at[i])
#        plt.plot(training_data[i][0], 'r')
#        plt.show()
#        plt.plot(training_data[i][1], 'g')
#        plt.show()
#        plt.plot(training_data[i][2], 'b')
#        plt.show()
#        plt.plot(training_data[i][3], 'y')
#        plt.show()

i = 0
for data in range(training_data.size):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(10, sharex=True, figsize=(20, 20))
    ax0.plot(training_data[data][0], 'r')
    ax0.set_title(training_labels.at[data])
    ax1.plot(training_data[data][1], 'r')
    ax2.plot(training_data[data][2], 'r')
    ax3.plot(training_data[data][3], 'r')
    ax4.plot(training_data[data][4], 'r')
    ax5.plot(training_data[data][5], 'r')
    ax6.plot(training_data[data][6], 'r')
    ax7.plot(training_data[data][7], 'r')
    ax8.plot(training_data[data][8], 'r')
    ax9.plot(training_data[data][9], 'r')
    path = "dataset/" + training_labels.at[data] + "/" + str(i)
    i += 1
    fig.savefig(path)
    plt.close(fig)
