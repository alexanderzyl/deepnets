from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizer_v2.gradient_descent import SGD


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


class ShallowModel:
    def __init__(self):
        self._net = None
        self.model_path = '/Users/aliaksandrzyl/Desktop/models/shallow'

    def create_model(self, num_classes):
        print("Compiling model...")
        opt = SGD(learning_rate=0.01)
        self._net = _create_net(width=32, height=32, depth=3, classes=num_classes)
        self._net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    @property
    def net(self):
        return self._net

    def load_model(self):
        from keras.models import load_model as _load
        self._net = _load(self.model_path)

    def save_model(self):
        self._net.save(self.model_path)
