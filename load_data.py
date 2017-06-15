from sklearn import cross_validation
from scipy.io import loadmat
def load_small_mnist(test_size):
	
	data = loadmat('ex3data1.mat')
	X, y = data['X'], data['y']
	y = y.reshape(X.shape[0], )
	y = y - 1
	return cross_validation.train_test_split(X, y, test_size=test_size)