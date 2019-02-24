from math import pi, atan, atan2, asin
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Maths

def count_local_maximum(array) :
#    plt.plot(array)
    c = 0
    for i in range(1, len(array) - 1) :
        if ((array[i] - array[i - 1]) * (array[i + 1] - array[i]) < 0) :
            c += 1
#    print(c)
#    plt.show()
    return c

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

def convolution_smooth(array) :
    r = np.convolve(array, [6/66, 24/66, 36/66, 24/66, 6/66], mode='valid')
    return r

# Utilities for models

#Prints accuracy values of models 
def accuracy_test(y_test, y_pred):
    count_misclassified = (y_test != y_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:.2f}'.format(accuracy))

def accuracy_average(classifier, X, y, nbTests) :
    score = cross_val_score(classifier, X, y, cv=nbTests)
    print("Score : ", score, " - ", np.mean(score))
    
def accuracy_average2(classifier, X, y) :
    score = cross_val_score(classifier, X, y, cv=8)
    print("Score : ", score, " - ", np.mean(score))

#Creates .cvs output file
def CSVOutput(y_kaggle, labelencoder):
    output = labelencoder.inverse_transform(y_kaggle)
    file = open("../submission.csv", "w+")
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

# Print cv
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=15):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_data = plt.cm.Paired
    cmap_data2 = plt.cm.tab20c
    cmap_cv = plt.cm.coolwarm
    
    indices = np.lexsort((group, y))
    group = group[indices]   
    X = X[indices]
    y = y[indices]    

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data2)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def conf_matrix(y_test, y_pred, normalize=True, cmap=plt.cm.Blues):
    labels = ["hard_tiles",
              "soft_pvc",
              "wood",
              "fine_concrete",
              "carpet",
              "concrete",
              "hard_tiles_lspace",
              "tiled",
              "soft_tiles"]

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,7))
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
        print("Normalized confusion matrix")
    else:
        title = 'Confusion matrix, without normalization'
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()