from keras.layers import Dense
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from callbacks import oneHot, CACallback


class PCANN:
    def __init__(self, layers=[400, 25, 10]):
        self.layers = layers
        self.model = Sequential()

    def fit(self, X_test, y_test, X_train, y_train, epochs=200, batch_size=10):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        activ = [X_train]
        for i in range(1, len(self.layers) - 1):
            activ.append(scaler.fit_transform(PCA(n_components=self.layers[i]).fit(X_train).transform(X_train)))
        activ.append(oneHot(y_train))

        NN_layers = []

        for i in range(len(self.layers) - 1):
            model = Sequential()
            model.add(Dense(self.layers[i + 1], input_dim=self.layers[i], activation='sigmoid'))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(activ[i], activ[i + 1], epochs=30, batch_size=10, verbose=0)
            NN_layers.append(model)

        for i in range(len(self.layers) - 1):
            self.model.add(Dense(self.layers[i + 1],
                                 input_dim=self.layers[i],
                                 activation='sigmoid',
                                 weights=NN_layers[i].layers[0].get_weights()))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        ca_callback = CACallback(X_test, oneHot(y_test), X_train, oneHot(y_train))
        self.model.fit(X_train, oneHot(y_train), epochs=epochs, batch_size=batch_size, verbose=0,
                       callbacks=[ca_callback])
        return ca_callback.get_data()