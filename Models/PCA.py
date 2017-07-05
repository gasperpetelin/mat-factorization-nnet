from keras.layers import Dense
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from callbacks import oneHot, CACallback


class PCANN:
    def __init__(self, layers=[400, 25, 10]):
        self.model = Sequential()
        self.layers = layers
        self.sigmoid_range_scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, X_test, y_test, X_train, y_train, epochs=200, batch_size=10):

        #Layerwise data features
        layerwise_activations = [X_train]
        for i in range(1, len(self.layers) - 1):
            pca_reduced_data = PCA(n_components=self.layers[i]).fit(X_train).transform(X_train)
            scaled_data = self.sigmoid_range_scaler.fit_transform(pca_reduced_data)
            layerwise_activations.append(scaled_data)
        layerwise_activations.append(oneHot(y_train))

        #Network mappings from i to i+1 layer
        mapping_neural_networks = []
        for i in range(len(self.layers) - 1):
            mapping_neural_network_model = Sequential()
            #Add 1 layer from input to output
            mapping_neural_network_model.add(Dense(self.layers[i + 1], input_dim=self.layers[i], activation='sigmoid'))
            mapping_neural_network_model.compile(loss='mean_squared_error', optimizer='adam')
            mapping_neural_network_model.fit(layerwise_activations[i], layerwise_activations[i + 1], epochs=30, batch_size=10, verbose=0)
            mapping_neural_networks.append(mapping_neural_network_model)

        #Build model from calculated weights
        for i in range(len(self.layers) - 1):
            self.model.add(Dense(self.layers[i + 1],
                                 input_dim=self.layers[i],
                                 activation='sigmoid',
                                 weights=mapping_neural_networks[i].layers[0].get_weights()))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        # Generate callback for classification accuracy
        ca_callback = CACallback(X_test, oneHot(y_test), X_train, oneHot(y_train))
        self.model.fit(X_train, oneHot(y_train),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=0,
                       callbacks=[ca_callback])
        # Return (train, test) accuracy tuple
        return ca_callback.get_data()