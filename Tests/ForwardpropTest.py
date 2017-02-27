import unittest
from ActivationFunctions import *
from NNet import *
from numpy import genfromtxt

class NNetTest(unittest.TestCase):
    def test_forwardArtificialNumbers(self):
        si = SigmoidActivationFunction()
        net = NNet(2, [3], 1, si, False)
        X = np.array([[3,5], [5, 1]])
        W1 = np.array([[1.3, -0.7, 0.9],[-2, 1.3, 0.4]])
        W2 = np.array([2,0.3,0.7]).transpose()
        (r, alist) = net.forwardpropagation(X, [W1, W2])
        #print(r, "NEKAJ")
        self.assertTrue(np.allclose(r, np.array([0.72998, 0.93719])))

    def test_forwardDigits(self):
        w1 = genfromtxt('..\\Data\\theta1.csv', delimiter=',')
        w2 = genfromtxt('..\\Data\\theta2.csv', delimiter=',')
        data = genfromtxt('..\\Data\\data.csv', delimiter=',')
        correct = genfromtxt('..\\Data\\correct.csv', delimiter=',')
        si = SigmoidActivationFunction()
        net = NNet(400, [3], 10, si, True)
        (pred, alist) = net.forwardpropagation(data, [w1.transpose(), w2.transpose()])
        #print(pred.shape, pred[0])
        Y = np.zeros((5000,10))
        for i in range(0, 5000):
            Y[i, int(correct[i]-1)] = 1
        self.assertAlmostEqual(net.error(pred, Y),0.287629, places=5)

    def test_forwardDigitsRegularized(self):
        w1 = genfromtxt('..\\Data\\theta1.csv', delimiter=',')
        w2 = genfromtxt('..\\Data\\theta2.csv', delimiter=',')
        data = genfromtxt('..\\Data\\data.csv', delimiter=',')
        correct = genfromtxt('..\\Data\\correct.csv', delimiter=',')
        si = SigmoidActivationFunction()
        net = NNet(400, [3], 10, si, True, 1)
        (pred, alist) = net.forwardpropagation(data, [w1.transpose(), w2.transpose()])
        #print(pred.shape, pred[0])
        Y = np.zeros((5000,10))
        for i in range(0, 5000):
            Y[i, int(correct[i]-1)] = 1
        self.assertAlmostEqual(net.error(pred, Y, [w1, w2], 1),0.383770, places=5)

    def test_backpropagation(self):
        w1 = genfromtxt('..\\Data\\theta1.csv', delimiter=',')
        w2 = genfromtxt('..\\Data\\theta2.csv', delimiter=',')
        data = genfromtxt('..\\Data\\data.csv', delimiter=',')
        correct = genfromtxt('..\\Data\\correct.csv', delimiter=',')
        si = SigmoidActivationFunction()
        net = NNet(400, [3], 10, si, True, 1)
        Y = np.zeros((5000,10))
        for i in range(0, 5000):
            Y[i, int(correct[i]-1)] = 1
        pred = net.backpropagation(data, [w1.transpose(), w2.transpose()], Y)
        print(w1.shape, w2.shape)
        print("------", pred[0].shape,pred[1].shape)
        #print(pred.shape, pred[0])

        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
