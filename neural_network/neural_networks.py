from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Reshape, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM

from keras import backend as K
K.set_image_dim_ordering('th')

def basic(input_size, output_size) :
    hidden_layer_size = int(sqrt(input_size * output_size))
    
    print("Input size : {}".format(input_size))
    print("Output size : {}".format(output_size))
    print("Hidden layer size : {}".format(hidden_layer_size))
    
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=input_size, units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=hidden_layer_size, kernel_initializer='uniform'))
    classifier.add(Dropout(0.2))
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

def convolutional_VGGinspired(input_size, output_size) :
    model = Sequential()
    model.add(Reshape((10, 128), input_shape=(1, 10, 128)))
    #model.add(Flatten(input_shape=(1, 10, 128)))

    model.add(Conv1D(2, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(2, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, strides=2))
    
    model.add(Conv1D(4, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(4, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Flatten())
    model.add(Dense(400, activation='relu', name='dense_A'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='sigmoid', name='dense_B'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def convolutional2D_random(input_size, output_size) :
    ''' Convolutional network requires the raveller extractor + reshaped in (1, 10, 128) '''
    import random

    nbfeatures_layer1 = int(random.random() * 150 + 20)
    nbfeatures_layer2 = int(nbfeatures_layer1 * 2 + random.random() * 100)
    size_dense1 = int(random.random() * 1500 + 200)
    kernel_width = 2 + int(random.random() * 18)
    prop_dropout1 = random.random() / 2 + 0.1
    prop_dropout2 = random.random() / 2 + 0.1

    print("Conv1: {}\nConv2: {}\nDropOut1: {}\nDropOut2: {}\nDense: {}".format(nbfeatures_layer1, nbfeatures_layer2, prop_dropout1, prop_dropout2, size_dense1))

    model = Sequential()
#    model.add(Reshape((1,10, 128), input_shape=(1, 10, 128)))
    model.add(Conv2D(nbfeatures_layer1, kernel_size=(10, kernel_width), strides=(1, 1), activation='relu', input_shape=(1, 10, 128), data_format = 'channels_first'))
    model.add(MaxPooling2D(pool_size=(1, 5)))
    model.add(Conv2D(nbfeatures_layer2, kernel_size=(1, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(prop_dropout1))
    model.add(Dense(size_dense1, activation='relu'))
    model.add(Dropout(prop_dropout2))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def convolutional_VGG(input_size, output_size) :
    from keras.applications import VGG16
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(3, 10, 128))
    last = vgg.output
    x = Flatten()(last)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(output_size, activation='softmax')(x)
    model = Model(initial_model.input, preds)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def recurrent(input_shape, output_size) :
    ''' Recurrent network requires the RNN extractor '''
    model = Sequential()
    model.add(LSTM(units=100, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
