import numpy as np
import math
import matplotlib.pyplot as plt
from utilities import tools

# Usefull functions

def is_croissant(array, treshold) :
    minimum_tolerated = array[0] - treshold
    local = array[0]
    i = 1
    while local > minimum_tolerated and i < len(array) :
        if (local - minimum_tolerated) > treshold :
            minimum_tolerated = local - treshold
        local = array[i]
        i+=1
    return 1 if i == len(array) else 0

def is_decroissant(array, treshold) :
    maximum_tolerated = array[0] + treshold
    local = array[0]
    i = 1
    while local < maximum_tolerated and i < len(array) :
        if (maximum_tolerated - local) > treshold :
            maximum_tolerated = local + treshold
        local = array[i]
        i+=1
    return 1 if i == len(array) else 0

def is_monotonous(array, treshold) :
    #print( 1 if is_decroissant(array, treshold) or is_croissant(array, treshold) else 0 )
    #plt.plot(array)
    #plt.show()
    return 1 if is_decroissant(array, treshold) or is_croissant(array, treshold) else 0

def count_local_maximum(array) :
#    plt.plot(array)
    c = 0
    for i in range(1, len(array) - 1) :
        if ((array[i] - array[i - 1]) * (array[i + 1] - array[i]) < 0) :
            c += 1
#    print(c)
#    plt.show()
    return c

# Extractors

def raveller(X_raw, size) :
    X = np.zeros((size, 1280))
    for i in range(size) :
        X[i] = np.ravel(X_raw[i])
    return X

def averager(X_raw, size) :
    X = np.zeros((size, 10))
    for i in range(size) :
        for j in range(10) :
            X[i, j] = np.mean(X_raw[i, j])
    return X

def deviationer(X_raw, size) :
    X = np.zeros((size, 20))
    for i in range(size) :
        for j in range(10) :
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = np.std(X_raw[i, j])
    return X

def deviationer_plus(X_raw, size) :
    X = np.zeros((size, 70))
    for i in range(size) :
        for j in range(10) :
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = np.std(X_raw[i, j])
            X[i, j+20] = X_raw[i, j].max()
            X[i, j+30] = X_raw[i, j].min()
            X[i, j+40] = count_local_maximum(tools.convolution_smooth(X_raw[i, j]))
    return X

def deviationer_monotonous(X_raw, size) :
    X = np.zeros((size, 50))
    for i in range(size) :
        for j in range(10) :
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = np.std(X_raw[i, j])
            X[i, j+20] = X_raw[i, j].max()
            X[i, j+30] = X_raw[i, j].min()
            treshold = ( X_raw[i, j].max() - X_raw[i, j].min() ) / 4
            X[i, j+40] = is_monotonous(X_raw[i, j], treshold)
    return X

def euler_angles(X_raw, size):
#    for i in range(128):
#        for j in range(size):
#            X_raw[j][0][i], X_raw[j][1][i], X_raw[j][2][i] = tools.quaternionToEulerAngles(X_raw[j][0][i], X_raw[j][1][i], X_raw[j][2][i], X_raw[j][3][i])
#            for k in range(3, 9):
#                X_raw[j][k][i] = X_raw[j][k + 1][i]
#            np.delete(X_raw[j], 9, 0)
#    X = np.zeros((size, 36))
#    for i in range(size) :
#        for j in range(9) :
#            X[i, j] = np.mean(X_raw[i, j])
#            X[i, j+9] = np.std(X_raw[i, j])
#            X[i, j+18] = X_raw[i, j].max()
#            X[i, j+27] = X_raw[i, j].min()
#    return X
    X_quater = deviationer_plus(X_raw, size)
    X = np.zeros((size, 49))
    for i in range(size): 
        euler = tools.quaternionToEulerAngles(X_quater[i][0], X_quater[i][1], X_quater[i][2], X_quater[i][3])
        for j in range(3):
            X[i][j] = euler[j]
        for j in range(3, 49):
            X[i][j] = X_quater[i][j + 1]
    return X

def no_orientation(X_raw, size):
    X = np.zeros((size, 30))
    for i in range(size) :
        for j in range(6) :
            X[i, j] = np.mean(X_raw[i, j + 4])
            X[i, j+6] = np.std(X_raw[i, j + 4])
            X[i, j+12] = X_raw[i, j + 4].max()
            X[i, j+18] = X_raw[i, j + 4].min()
            X[i, j+24] = count_local_maximum(tools.convolution_smooth(X_raw[i, j + 4]))
    return X

def get_all_extractors() :
    return [('raveller', raveller),
            ('averager', averager),
            ('deviationer', deviationer),
            ('deviationer_plus', deviationer_plus),
            ('deviationer_monotonous', deviationer_monotonous),
            ('euler_angles', euler_angles)]

def features_extractors_benchmark() :
    import loaders
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier

    extractors = [('raveller', raveller), 
                  ('averager', averager),
                  ('deviationer', deviationer),
                  ('deviationer_plus', deviationer_plus),
                  ('deviationer_monotonous', deviationer_monotonous),
                  ('euler_angles', euler_angles)]

    for extractor in extractors :
        X_train, y_train, X_test, y_test, le = loaders.load_for_train(0.20, extractor[1])
        lda = LinearDiscriminantAnalysis()
        etc = ExtraTreesClassifier(1000)

        lda.fit(X_train, y_train)
        etc.fit(X_train, y_train)

        print(extractor[0])
        print("LDA")
        y_pred = lda.predict(X_test)
        tools.accuracy_test(y_test, y_pred)
        print("ETC")
        y_pred = etc.predict(X_test)
        tools.accuracy_test(y_test, y_pred)
        print()

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   features_extractors_benchmark()
