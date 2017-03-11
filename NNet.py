import numpy as np
from scipy import optimize


class SigmoidActivationFunction:
    
    @staticmethod
    def value(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z):
        sig = SigmoidActivationFunction.value(z)
        return sig * (1 - sig)
    
    
class CrossEntropyCostFunction:

    @staticmethod
    def cost(actual, predicted, numberOfExamples):
        return np.sum(-actual * np.log(predicted).T - (1 - actual) * np.log(1 - predicted).T) / numberOfExamples    
    
    @staticmethod
    def regulazation(weightsList, lambdaFactor, numberOfelements):
        cost = 0
        if lambdaFactor == 0:
            for w in weightsList:
                cost +=np.dot(w, w)
            return (lambdaFactor/(2*numberOfelements))*cost
        return cost
    
    @staticmethod
    def delta(actual, predicted, activationFunction):
        return predicted-actual



class NN_1HL:
    #opti_method='TNC' BFGS
    def __init__(self, layerSizes = [10,5,3], reg_lambda=0, epsilon_init=0.12, 
                 hidden_layer_size1=50,hidden_layer_size2 = 30,  opti_method='TNC', maxiter=500,
                 activationFunction = SigmoidActivationFunction, costFunction = CrossEntropyCostFunction):
        self._reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size1 = hidden_layer_size1
        self.hidden_layer_size2 = hidden_layer_size2
        self._method = opti_method
        self._maxiter = maxiter
        self._costFunction = costFunction
        self._activationFunction = activationFunction
        self._layerSizes = layerSizes
    
    def rand_init(self, l_in, l_out):
        return np.random.randn(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init
    
    def packWeights(self, weightsList):
        return np.concatenate([w.ravel() for w in weightsList])
    
    def unpackWeightsAlgorithm(self, weights, layerSizes):
        requredLen = sum([y*x for y,x in layerSizes])
        if requredLen == len(weights):
            start = 0
            returnList = []
            for y,x in layerSizes:
                returnList.append(weights[start:start+y*x].reshape((y,x)))
                start +=y*x
            return returnList
        else:
            print("Weights sizes mismatch,", requredLen, "weights requred")
            
    def unpackWeights(self, thetas, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels):
        sizes = [(hidden_layer_size1, input_layer_size + 1),
                 (hidden_layer_size2, hidden_layer_size1 + 1),  
                 (num_labels, hidden_layer_size2 + 1)]
        ls = self.unpackWeightsAlgorithm(thetas, sizes)
        return ls[0], ls[1], ls[2]
    
    def _forward(self, X, t1, t2, t3):        
        a1 = self.addOnes(X)

        z2 = np.dot(t1, a1.T)
        a2 = self._activationFunction.value(z2)
        a2 = self.addOnes(a2.T)
        
        
        z3 = np.dot(t2, a2.T)
        a3 = self._activationFunction.value(z3)
        a3 = self.addOnes(a3.T)
        
        z4 = np.dot(t3, a3.T)
        a4 = self._activationFunction.value(z4)
        
        return a1, z2, a2, z3, a3, z4, a4
    
    
    def variableSetup(self, thetas, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, X, y):
        t1, t2, t3 = self.unpackWeights(thetas, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels)
        m = X.shape[0]
        Y = np.eye(num_labels)[y]
        return (t1, t2, t3, m, Y)
        
    def removeBiasesFromWeightMatrices(self, listOfWeights):
        return [w[:, 1:] for w in listOfWeights]
    
    def function(self, thetas, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, X, y, reg_lambda):
        t1, t2, t3, m, Y = self.variableSetup(thetas, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, X, y)
        
        a1, z2, a2, z3, a3, z4, a4 = self._forward(X, t1, t2, t3)
        
        J = self._costFunction.cost(Y, a4, m)
        reg = self._costFunction.regulazation(self.removeBiasesFromWeightMatrices([t1, t2]), self._reg_lambda, m)
        

        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        t3f = t3[:, 1:]

        
        
        si4 = self._costFunction.delta(Y, a4.T, self._activationFunction)
        si3 = (np.dot(si4, t3) * self._activationFunction.derivative(self.addOnes(z3.T)))[:, 1:]
        si2 = (np.dot(si3, t2) * self._activationFunction.derivative(self.addOnes(z2.T)))[:, 1:]

        d1 = np.dot(si2.T,a1);
        d2 = np.dot(si3.T,a2);
        d3 = np.dot(si4.T,a3);
        
        Theta1_grad = d1 / m
        Theta2_grad = d2 / m
        Theta3_grad = d3 / m
        
        if reg_lambda != 0:
            Theta1_grad[:, 1:] += (reg_lambda / m) * t1f
            Theta2_grad[:, 1:] += (reg_lambda / m) * t2f
            Theta3_grad[:, 1:] += (reg_lambda / m) * t3f
        
        
        return (J + reg, self.packWeights([Theta1_grad, Theta2_grad, Theta3_grad]))
    
    def addOnes(self, x):
        ys,xs = x.shape
        z = np.ones((ys,1))
        return np.concatenate((z, x), axis=1)
    
    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))
        
        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size1)
        theta2_0 = self.rand_init(self.hidden_layer_size1, self.hidden_layer_size2)
        theta3_0 = self.rand_init(self.hidden_layer_size2, num_labels)
        
        thetas0 = self.packWeights([theta1_0, theta2_0, theta3_0])
        
        options = {'maxiter': self._maxiter}
        _res = optimize.minimize(self.function, thetas0, jac=True, method=self._method, 
                                 args=(input_layer_size, self.hidden_layer_size1, self.hidden_layer_size2, num_labels, X, y, self._reg_lambda), options=options)
        
        self.t1, self.t2, self.t3 = self.unpackWeights(_res.x, input_layer_size, self.hidden_layer_size1, self.hidden_layer_size2, num_labels)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(0)
    
    def predict_proba(self, X):
        _, _, _, _, _, _, h = self._forward(X, self.t1, self.t2, self.t3)
        return h




import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score


from scipy.io import loadmat
data = loadmat('ex3data1.mat')
np.random.seed(40)
X, y = data['X'], data['y']
y = y.reshape(X.shape[0])
y = y - 1  # Fix notation # TODO: Automaticlly fix that on the class

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

nn = NN_1HL(layerSizes=[400, 25,10], maxiter=300, reg_lambda=6)

nn.fit(X_train, y_train)

print(accuracy_score(y_test, nn.predict(X_test)))