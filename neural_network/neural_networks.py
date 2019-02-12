import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Reshape, GlobalAveragePooling1D, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM

from keras import backend as K
K.set_image_dim_ordering('th')

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
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(output_size, activation='softmax'))
    model_m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_m

def convolutional2D(input_size, output_size) :
    ''' Convolutional network requires the raveller extractor + reshaped in (1, 10, 128) '''
    model = Sequential()
#    model.add(Reshape((1,10, 128), input_shape=(1, 10, 128)))
    model.add(Conv2D(50, kernel_size=(10, 20), strides=(1, 1), activation='relu', input_shape=(1, 10, 128), data_format = 'channels_first'))
    model.add(MaxPooling2D(pool_size=(1, 5)))
    model.add(Conv2D(100, kernel_size=(1, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def recurrent(input_shape, output_size) :
    ''' Recurrent network requires the RNN extractor '''
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
