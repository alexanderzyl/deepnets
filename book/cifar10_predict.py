from keras.datasets import cifar10
import numpy as np
import cv2

from book.shallow_model import ShallowModel

print("Loading model...")
model = ShallowModel()
model.load_model()
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Preparing Data...")
(_, (testX, testY)) = cifar10.load_data()
testX = testX.astype('float') / 255.

np.random.seed(109)
idx = np.random.choice(testX.shape[0], 32)

randomX = np.take(testX, idx, axis=0)
randomY = np.take(testY, idx, axis=0)

print("Predicting...")
predictions = model.net.predict(randomX)

print("Show images")
for image, p in zip(randomX, predictions.argmax(axis=1)):
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    cv2.putText(resized, label_names[p], (10, 30), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
    cv2.imshow("image", resized)
    cv2.waitKey(0)
