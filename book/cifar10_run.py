from keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from book.shallow_model import MODELS_SHALLOW_PATH

# from book.shallow_model import create_model as get_model
from book.shallow_model import load_model as get_model

print("Preparing Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.
testX = testX.astype('float') / 255.

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = get_model(len(label_names))

print("Training...")
H = model.fit(trainX, trainY, batch_size=32, epochs=4, verbose=1)

print('Evaluating...')
predictions = model.predict(testX, batch_size=32)

cr = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)
print(cr)

model.save(MODELS_SHALLOW_PATH)
