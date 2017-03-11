import numpy as np

class SigmoidActivationFunction:
    def value(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return np.multiply(self.value(x), 1-self.value(x))