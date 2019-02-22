print("Importing librairies...")
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utilities import loaders
from utilities import feature_extractors
from utilities import tools
from mixupgenerator.mixup_generator import MixupGenerator
import neural_networks

print("Done.")

test_size = 0.20
batch_size = 80
epochs = 25
alpha = 0.4

i = 0

print("Loading dataset")

extractor = feature_extractors.raveller
indices_generator, le = loaders.load_for_train_groups(test_size, extractor)

print("Done.")

kaggle_classification = int(input("Classification for Kaggle ? (1 or 0) : "))

if not kaggle_classification :
    print("\n ▁▃▅▇ SPLIT {} ▇▅▃▁ \n".format(i))
    i += 1

    print("▶ Loading training data & preprocessing ◀")
    X_train, y_train, X_test, y_test, lb = loaders.load_for_train_keras(test_size, extractor)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
      
    X_train = X_train.reshape((len(X_train), 1, 10, 128))
    X_test = X_test.reshape((len(X_test), 1, 10, 128))
    training_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=alpha)()
       
    print("▶ Building the neural network ◀")
    input_size = 1*128*10
    output_size = 9
    #classifier = neural_networks.basic(input_size, output_size)
    classifier = neural_networks.convolutional_VGG(input_size, output_size)

    print("▶ Training ◀")
    classifier.fit(X_train, y_train, batch_size=10, epochs=epochs)
    #classifier.fit_generator(generator=training_generator, steps_per_epoch=X_train.shape[0] // batch_size, validation_data=(X_test, y_test), epochs=epochs, verbose=1)

    print("▶ Evaluating ◀")
    y_pred = lb.inverse_transform(tools.max_one_hot(classifier.predict(X_test)), 0.5)
    y_test = lb.inverse_transform(y_test, 0.5)
    tools.accuracy_test(y_test, y_pred)
    tools.conf_matrix(y_test, y_pred)
else :
    print("▶ Loading training data & preprocessing ◀")
    X, y, X_kaggle, lb = loaders.load_for_kaggle_keras(extractor)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_kaggle = sc.transform(X_kaggle)
      
    X = X.reshape((len(X), 1, 10, 128))
    X_kaggle = X_kaggle.reshape((len(X_kaggle), 1, 10, 128))
    training_generator = MixupGenerator(X, y, batch_size=batch_size, alpha=alpha)()
    
    print("▶ Building the neural network ◀")
    classifier = neural_networks.convolutional2D(1280, 9)

    print("▶ Training ◀")
    classifier.fit_generator(generator=training_generator, steps_per_epoch=X.shape[0] // batch_size, epochs=epochs, verbose=1)
	
    print("▶ Evaluating Kaggle Data ◀")
    y_pred = lb.inverse_transform(tools.max_one_hot(classifier.predict(X_kaggle)), 0.5)
    
    print("▶ Generating submission file ◀")
    tools.CSVOutput(y_pred)
