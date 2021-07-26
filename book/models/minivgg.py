from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizer_v2.gradient_descent import SGD

from book.models.any_model import AnyModel


def _create_net(height, width, depth, classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height, width, depth)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model


class MiniVGG(AnyModel):
    def __init__(self):
        super().__init__('/Users/aliaksandrzyl/Desktop/models/minivgg')

    def create_model(self, num_classes):
        print("Compiling model...")
        opt = SGD(learning_rate=0.01)
        self._net = _create_net(width=32, height=32, depth=3, classes=num_classes)
        self._net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
