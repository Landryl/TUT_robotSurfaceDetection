import numpy as np
import math
import matplotlib.pyplot as plt

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
    X = np.zeros((size, 40))
    for i in range(size) :
        for j in range(10) :
            X[i, j] = np.mean(X_raw[i, j])
            X[i, j+10] = np.std(X_raw[i, j])
            X[i, j+20] = X_raw[i, j].max()
            X[i, j+30] = X_raw[i, j].min()
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

def get_all_extractors() :
    return [('raveller', raveller),
            ('averager', averager),
            ('deviationer', deviationer),
            ('deviationer_plus', deviationer_plus),
            ('deviationer_monotonous', deviationer_monotonous)]

def features_extractors_benchmark() :
    import loaders
    import tools
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    extractors = [('raveller', raveller), 
                  ('averager', averager),
                  ('deviationer', deviationer),
                  ('deviationer_plus', deviationer_plus),
                  ('deviationer_monotonous', deviationer_monotonous)]

    for extractor in extractors :
        X_train, y_train, X_test, y_test, le = loaders.load_for_train(0.20, extractor[1])
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        print(extractor[0])
        tools.accuracy_test(y_test, y_pred)
        print()
