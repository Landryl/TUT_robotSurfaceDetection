from math import pi, atan, atan2, asin
import numpy as np
from sklearn.metrics import accuracy_score

# Maths

def quaternionToEulerAngles(x, y, z, w) :
    # X axis rotation
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    # Y axis rotation
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1 :
    	pitch = pi/2 * (sinp/sinp) # use 90 degrees if out of range
    else :
        pitch = asin(sinp)

    # Z axis rotation
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


# Utilities for models

#Prints accuracy values of models 
def accuracy_test(y_test, y_pred):
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))


#Creates .cvs output file
def CSVOutput(y_kaggle, labelencoder):
    output = labelencoder.inverse_transform(y_kaggle)
    file = open("submission.csv", "w+")
    file.write("# Id,Surface\n")
    for i in range(output.size):
        line = str(i) + "," + output[i] + "\n"
        file.write(line)
    file.close()
    
# Print total data by groups
def print_total_groups(dataset):
    groups_labeled = dataset.iloc[:, [1,2]].values
    
    total_group_numbers, indices, total_counts = np.unique(groups_labeled[:, 0], return_index=True, return_counts=True)
    total_labels = groups_labeled[indices, 1]
    
    print("Total data (group_number - counts - labels):\n", np.asarray((total_group_numbers, total_counts, total_labels)).T)
    print()
    
    
# Printing groups repartition in test and train data
def print_train_test_repartition(dataset, train_index, test_index):
    groups_labeled = dataset.iloc[:, [1,2]].values
    groups_train, groups_test = groups_labeled[train_index], groups_labeled[test_index]
    
    train_group_numbers, train_counts = np.unique(groups_train[:, 1], return_counts=True)
    test_group_numbers, test_counts = np.unique(groups_test[:, 1], return_counts=True)

    print("Training data (labels - count):\n", np.asarray((train_group_numbers, train_counts)).T)
    print()
    print("Testing data (labels - count):\n", np.asarray((test_group_numbers, test_counts)).T)
    print()
