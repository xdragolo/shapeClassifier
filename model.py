from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.regularizers import l2


def getNN(input_dim, n1,l):
    model = Sequential()
    model.add(Dense(n1, input_dim =input_dim,activation='sigmoid', activity_regularizer=regularizers.l2(l)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
