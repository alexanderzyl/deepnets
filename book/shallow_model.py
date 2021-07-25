from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizer_v2.gradient_descent import SGD

MODELS_SHALLOW_PATH = '/Users/aliaksandrzyl/Desktop/models/shallow'


def create_net(height, width, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


def create_model(num_classes):
    print("Compiling model...")
    opt = SGD(learning_rate=0.01)
    model = create_net(width=32, height=32, depth=3, classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def load_model(_):
    from keras.models import load_model as _load
    return _load(MODELS_SHALLOW_PATH)
