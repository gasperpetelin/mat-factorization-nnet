import numpy as np

class NNet:
    def __init__(self, inputLayerSize, hiddenLayerSizes, outputLayerSize, activationFunction, includeBias = True, regularization = 0):
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSizes = hiddenLayerSizes
        self.outputLayerSize = outputLayerSize
        self.activationFunction = activationFunction
        self.includeBias = includeBias

    def forwardpropagation(self, X, weightMatrices):
        x = X
        returnList = []

        for w in weightMatrices:
            if self.includeBias:
                ys,xs = x.shape
                z = np.ones((ys,1))
                x = np.concatenate((z, x), axis=1)
            returnList.append(x)
            x = self.activationFunction.value(np.dot(x, w))
        return (x, returnList)

    def error(self, X, Y, weightsList = None, lambdaFactor = 0):
        ys, xs = X.shape
        reg = 0
        if weightsList is not None:
            for w in weightsList:
                reg += np.sum(np.power(w[:, 1:], 2))
        reg = (lambdaFactor * reg) / (2 * ys)
        return (np.sum(np.multiply(-Y, np.log(X)) - np.multiply(1-Y, np.log(1-X)))/ys) + reg

    def backpropagation(self, X, weightMatrices, Y):
        (pred, A) = self.forwardpropagation(X, weightMatrices)
        #print(len(A))

        a1 = A[0]
        a2 = A[1]
        a3 = A[2]


        si3 = a3-Y

        ys,xs = a2.shape
        z2 = np.concatenate((np.ones((ys,1)), a2) , axis=1)

        #neki = np.dot(si3, weightMatrices[1].transpose())


        si2 = np.multiply(np.dot(si3, weightMatrices[1].transpose()),self.activationFunction.derivative(z2))

        #print(si2.shape)
        si2 = si2[:, 1:]
        #print("......", si2.shape)
        #print(si2.shape)

        #print(si2.transpose().shape, "si2shape")
        print(a1.shape, "a1")
        d1 = np.dot(si2.transpose(), a1);
        d2 = np.dot(si3.transpose(), a2);
        print(d1.shape, "d1")
        print(d2.shape, "d2")

        return [d1/ys,d2/ys]





