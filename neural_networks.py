import keras
from keras.models import Sequential
from keras.layers import Dense

def basic(input_size, output_size) :
    hidden_layer_size = int((input_size + output_size) / 2)
    
    print("Input size : {}".format(input_size))
    print("Output size : {}".format(output_size))
    print("Hidden layer size : {}".format(hidden_layer_size))
    
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=input_size, units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dense(activation='softmax', units=output_size, kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier
