import numpy as np

def edvc(array_of_classifiers, X_test) :
    votes = [[0 for i in range(9)] for j in range(len(X_test))]
    y_preds = []
    for c in array_of_classifiers :
        y_preds.append(c.predict(X_test))
    for y_pred in y_preds :
        for i in range(y_pred.size) :
            votes[i][y_pred[i]] += 1
    results = []
    for (idx, vote) in enumerate(votes) :
        max_i = 0
        for i in range(len(vote)) :
            if vote[i] > vote[max_i] :
                max_i = i
        if vote[max_i] == 1 :
            print("egalité")
            results.append(y_preds[0][idx])
        else :
            results.append(max_i)
    return np.array(results)
