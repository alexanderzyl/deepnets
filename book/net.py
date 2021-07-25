from keras.layers import Activation, Flatten, Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv2D


class ShallowNet:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model