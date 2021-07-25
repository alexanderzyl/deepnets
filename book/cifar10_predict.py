from keras.datasets import cifar10
import numpy as np

from book.shallow_model import ShallowModel

print("Loading model...")
model = ShallowModel()
model.load_model()

print("Preparing Data...")
(_, (testX, testY)) = cifar10.load_data()
testX = testX.astype('float') / 255.

np.random.seed(101)
idx = np.random.choice(testX.shape[0], 32)

randomX = np.take(testX, idx, axis=0)
randomY = np.take(testY, idx, axis=0)

print("Predicting...")
predictions = model.net.predict(randomX)
print(testX)



