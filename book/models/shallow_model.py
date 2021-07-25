from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizer_v2.gradient_descent import SGD

from book.models.any_model import AnyModel


def _create_net(height, width, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (1, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


class ShallowModel(AnyModel):
    def __init__(self):
        super().__init__('/Users/aliaksandrzyl/Desktop/models/shallow')

    def create_model(self, num_classes):
        print("Compiling model...")
        opt = SGD(learning_rate=0.01)
        self._net = _create_net(width=32, height=32, depth=3, classes=num_classes)
        self._net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
