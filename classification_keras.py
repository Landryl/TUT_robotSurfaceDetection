print("Importing librairies...")
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense

import loaders
import feature_extractors
import tools

print("Done.")

test_size = 0.20
i = 0

print("Loading dataset")

extractor = feature_extractors.euler_angles
indices_generator, le = loaders.load_for_train_groups(test_size, extractor)
X, y, X_kaggle, le = loaders.load_for_kaggle(extractor)

print("Done.")

for train_index, test_index in indices_generator :
    print("\n ▁▃▅▇ SPLIT {} ▇▅▃▁ \n".format(i))
    i += 1
    
    print("▶ Loading training data & preprocessing ◀")
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
    X_train, y_train, X_test, y_test, lb = loaders.load_for_train_keras(test_size, extractor)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
   
    print("▶ Building the neural network ◀")
    input_size = len(X_train[0])
    output_size = 9
    hidden_layer_size = int((input_size + output_size) / 2)
    print("Input size : {}".format(input_size))
    print("Output size : {}".format(output_size))
    print("Hidden layer size : {}".format(hidden_layer_size))

    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=input_size, units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dense(activation='softmax', units=output_size, kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("▶ Training ◀")
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)    

    print("▶ Evaluating ◀")
    y_pred = lb.inverse_transform(classifier.predict(X_test), 0.5)
    y_test = lb.inverse_transform(y_test, 0.5)
    tools.accuracy_test(y_test, y_pred)

