import numpy as np
import tools

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

def euler_angles(X_raw, size):
    X_quater = deviationer_plus(X_raw, size)
    X = np.zeros((size, 36))
    for i in range(size): 
        euler = tools.quaternionToEulerAngles(X_quater[i][0], X_quater[i][1], X_quater[i][2], X_quater[i][3])
        for j in range(3):
            X[i][j] = euler[j]
        for j in range(3, 9):
            X[i][j] = X_quater[i][j + 1]
    return X

def get_all_extractors() :
    return [('raveller', raveller),
            ('averager', averager),
            ('deviationer', deviationer),
            ('deviationer_plus', deviationer_plus),
            ('euler_angles', euler_angles)]

def features_extractors_benchmark() :
    import loaders
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    extractors = [('raveller', raveller), 
                  ('averager', averager),
                  ('deviationer', deviationer),
                  ('deviationer_plus', deviationer_plus)]

    for extractor in extractors :
        X_train, y_train, X_test, y_test, le = loaders.load_for_train(0.20, extractor[1])
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        print(extractor[0])
        tools.accuracy_test(y_test, y_pred)
        print()
