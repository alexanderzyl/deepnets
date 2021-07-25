from keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from book.plot import plot_history

from book.models.shallow_model import ShallowModel

print("Preparing Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.
testX = testX.astype('float') / 255.

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = ShallowModel()
model.create_model(len(label_names))

print("Training...")
epochs = 20
H = model.net.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)
model.save_model()

print('Evaluating...')
predictions = model.net.predict(testX, batch_size=32)

cr = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)
print(cr)

plot_history(H.history, epochs)


