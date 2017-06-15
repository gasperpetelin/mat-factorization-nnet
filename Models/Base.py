from keras.layers import Dense, Activation
from keras.models import Sequential

from callbacks import *


class BaseNN:
    def __init__(self, layers=[400, 25, 10]):
        self.model = Sequential()
        for i in range(len(layers)-1):
            self.model.add(Dense(layers[i+1], input_dim=layers[i]))
            self.model.add(Activation('sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit(self, X_test, y_test, X_train, y_train, epochs = 200, batch_size = 10):
        ca_callback = CACallback(X_test, oneHot(y_test), X_train, oneHot(y_train))
        self.model.fit(X_train, oneHot(y_train), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[ca_callback])
        return ca_callback.get_data()
