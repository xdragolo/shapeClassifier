import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from model import getNN
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

stars = np.load('stars.npy')
circles = np.load('circles.npy')

X = np.concatenate((stars, circles), axis=0)
X = X / np.linalg.norm(X)

Y = np.concatenate((np.zeros(len(stars)), np.ones(len(circles))), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)


model = getNN(input_dim=X_train.shape[1], n1=400, l=0.001)
print(model.summary())
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2, batch_size=32)

accr = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

model.save('model')
# learning curves
print(history.history.keys())
plt.figure(figsize=(10, 3))
plt.subplot(111)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('./Figures/learningCurveLoss.png')
plt.show();

plt.subplot(122)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.savefig('./Figures/learningCurveAccuracy.png')
plt.show();
