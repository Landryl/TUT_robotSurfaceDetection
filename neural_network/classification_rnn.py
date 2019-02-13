print("Importing librairies...")
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utilities import loaders
from utilities import feature_extractors
from utilities import tools
from mixupgenerator.mixup_generator import MixupGenerator
import neural_networks
import numpy as np

print("Done.")

test_size = 0.20
batch_size = 60
epochs = 1
alpha = 0.4
input_shape = (128, 10)    # (timesteps, features)
output_size = 9


print("Loading dataset")

extractor = feature_extractors.RNN_extractor
indices_generator, le = loaders.load_for_train_groups(test_size, extractor)

print("Done.")

kaggle_classification = int(input("Classification for Kaggle ? (1 or 0) : "))

if not kaggle_classification :
    repeats = int(input("How many iterations ? "))
    print("▶ Loading training data & preprocessing ◀")
    X_train, y_train, X_test, y_test, lb = loaders.load_for_train_keras(test_size, extractor)
    #X_train, y_train, X_test, y_test, lb = loaders.load_for_train_keras_categorical(test_size, extractor)

    # Scaling
    sc = StandardScaler()
    for x in X_train:
        x = sc.fit_transform(x)
    for x in X_test:
        x = sc.fit_transform(x)

    # Mixup Generator
    #X_train = X_train.reshape((len(X_train), 1, 10, 128))
    #X_test = X_test.reshape((len(X_test), 1, 10, 128))
    #training_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=alpha)()
    training_generator = tools.generator(X_train, y_train, batch_size)

    scores = []
    for r in range(repeats):
        print("▶ Building the neural network ◀")
        classifier = neural_networks.recurrent(input_shape, output_size)

        print("▶ Training ◀")
        #classifier.fit(X_train, y_train, validation_data=[X_test, y_test],
        #              batch_size=10, epochs=1)
        classifier.fit_generator(generator=training_generator,
                                 steps_per_epoch=X_train.shape[0]//batch_size,
                                 validation_data=(X_test, y_test),
                                 epochs=epochs,
                                 verbose=1)

        print("▶ Evaluating ◀")
        score = classifier.evaluate(X_test, y_test, verbose=0)
        scores.append(score[1]*100)
        print('#%d: %.3f\n' % (r+1, score[1]))
        
        if repeats == 1:
            y_pred = lb.inverse_transform(tools.max_one_hot(classifier.predict(X_test)), 0.5)
            y_test = lb.inverse_transform(y_test, 0.5)
            tools.conf_matrix(y_test, y_pred)

    print("▶ Final results ◀")
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
else :
    print("▶ Loading training data & preprocessing ◀")
    X, y, X_kaggle, lb = loaders.load_for_kaggle_keras(extractor)

    # Scaling
    sc = StandardScaler()
    for x in X:
        x = sc.fit_transform(x)
    for x in X_kaggle:
        sc.transform(x)

    # Mixup Generator
    #X = X.reshape((len(X), 1, 10, 128))
    #X_kaggle = X_kaggle.reshape((len(X_kaggle), 1, 10, 128))
    #training_generator = MixupGenerator(X, y, batch_size=batch_size, alpha=alpha)()
    
    print("▶ Building the neural network ◀")
    classifier = neural_networks.recurrent(input_shape, output_size)

    print("▶ Training ◀")
    classifier.fit(X, y, batch_size=10, epochs=50)
    #classifier.fit_generator(generator=training_generator,
    #                         steps_per_epoch=X.shape[0],
    #                         batch_size=batch_size,
    #                         epochs=epochs,
    #                         verbose=1)

    print("▶ Evaluating Kaggle Data ◀")
    y_pred = lb.inverse_transform(tools.max_one_hot(classifier.predict(X_kaggle)), 0.5)

    print("▶ Generating submission file ◀")
    tools.CSVOutput(y_pred)
