from keras.datasets import cifar10
from keras.optimizer_v2.gradient_descent import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from book.net import ShallowNet

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.
testX = testX.astype('float') / 255.

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Compiling model...")
opt = SGD(learning_rate=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=len(labelNames))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Training...")
H = model.fit(trainX, trainY, batch_size=32, epochs=4, verbose=1)

print('Evaluating...')
predictions = model.predict(testX, batch_size=32)

cr = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)
print(cr)
