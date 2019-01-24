import numpy as np

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


def features_extractors_benchmark() :
    import loaders
    import tools
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
