import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Reshape, GlobalAveragePooling1D, Dropout

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

def convolutional(input_size, output_size) :
    ''' Convolutional network requires the raveller extractor '''
    model_m = Sequential()
    model_m.add(Reshape((128, 10), input_shape=(input_size,)))
    model_m.add(Conv1D(100, 10, activation='relu', input_shape=(128, 10)))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(output_size, activation='softmax'))
    model_m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_m

