import unittest
import numpy as np
from ActivationFunctions import *

class SigmoidTest(unittest.TestCase):
    def test_singleValue(self):
        f = SigmoidActivationFunction()
        self.assertAlmostEqual(f.value(0), 0.5)
        self.assertAlmostEqual(f.value(1), 0.7310585786300048)
        self.assertAlmostEqual(f.value(2), 0.8807970779778824)
        self.assertAlmostEqual(f.value(-13), 0.00000226032429)

    def test_vector(self):
        f = SigmoidActivationFunction()
        input = np.array([0, 1])
        output = np.array([0.5, 0.7310585786300048])
        self.assertTrue(np.allclose(f.value(input), output))

    def test_derivativeSignleValue(self):
        f = SigmoidActivationFunction()
        self.assertAlmostEqual(f.derivative(-1), 0.19661193324148)
        self.assertAlmostEqual(f.derivative(13), 2.26031918888760e-006)

    def test_derivativeVector(self):
        f = SigmoidActivationFunction()
        input = np.array([-1, 0 ,1 ,2])
        output = np.array([0.19661193324148, 0.25, 0.196611933241482, 0.104993585403507])
        print(f.derivative(input))
        self.assertTrue(np.allclose(f.derivative(input), output))

if __name__ == '__main__':
    unittest.main()
