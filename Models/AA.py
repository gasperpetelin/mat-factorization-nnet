from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from aa import AA
from callbacks import oneHot, CACallback


class AANN:
    def __init__(self, layers=[400, 25, 10], aa_iterations=15, regularizer_factor=0):
        self.model = Sequential()
        self.layers = layers
        self.aa_iterations = aa_iterations
        self.regularizer_factor = regularizer_factor

    def fit(self, X_test, y_test, X_train, y_train, epochs=200, batch_size=10):

        # Layerwise data features
        layerwise_activations = [X_train]
        for i in range(1, len(self.layers) - 1):
            aa = AA(X_train, self.layers[i])
            aa.factorize(self.aa_iterations)
            aa_reduced_data = aa.W
            layerwise_activations.append(aa_reduced_data)
        layerwise_activations.append(oneHot(y_train))

        # Network mappings from i to i+1 layer
        mapping_neural_networks = []
        for i in range(len(self.layers) - 1):
            mapping_neural_network_model = Sequential()
            # Add 1 layer from input to output
            mapping_neural_network_model.add(Dense(self.layers[i + 1],
                                                   input_dim=self.layers[i],
                                                   activation='sigmoid',
                                                   W_regularizer=l2(self.regularizer_factor)))
            mapping_neural_network_model.compile(loss='mean_squared_error', optimizer='adam')
            mapping_neural_network_model.fit(layerwise_activations[i], layerwise_activations[i + 1],
                                             epochs=30, batch_size=10, verbose=0)
            mapping_neural_networks.append(mapping_neural_network_model)

        # Build model from calculated weights
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