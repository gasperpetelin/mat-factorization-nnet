import keras
import matplotlib.pyplot as plt
import numpy as np


class CACallback(keras.callbacks.Callback):
    def __init__(self, X_test, y_test, X_train, y_train):
        self.X_test = X_test
        self.y_test = y_test
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.test_ca = []
        self.train_ca = []

    def on_epoch_begin(self, epoch, logs={}):
        test_loss, test_ca = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        train_loss, train_ca = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        
        self.test_ca.append(test_ca)
        self.train_ca.append(train_ca)
        
    def plot(self):
        plt.plot(self.test_ca, label="Test data")
        plt.plot(self.train_ca, label="Train data")
        plt.title("Classification accuracy")
        plt.ylabel('CA')
        plt.xlabel('Number of epochs')
        plt.ylim([0,1])
        plt.legend(loc='lower right')
        plt.show()
        
    def get_data(self):
        return (self.train_ca, self.test_ca)
        
class LossCallback(keras.callbacks.Callback):
    def __init__(self, X_test, y_test, X_train, y_train):
        self.X_test = X_test
        self.y_test = y_test
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.test_loss = []
        self.train_loss = []
        
    def on_epoch_begin(self, epoch, logs={}):
        test_loss, test_ca = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        train_loss, train_ca = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        
        self.test_loss.append(test_loss)
        self.train_loss.append(train_loss)
        
    def plot(self):
        plt.plot(self.test_loss, label="Test data")
        plt.plot(self.train_loss, label="Train data")
        plt.title("Loss")
        plt.ylabel('Loss')
        plt.xlabel('Number of epochs')
        plt.legend(loc='upper right')
        plt.show()
        
    def get_data(self):
        return (self.train_loss, self.test_loss)
        
class IterationCallback(keras.callbacks.Callback):
    def __init__(self, mod):
        self.mod = mod
        self.itr = 0
        
    def on_epoch_begin(self, epoch, logs={}):
        self.itr +=1
        if self.itr % self.mod==0:
            print(self.itr)
			
def oneHot(X):
    return np.eye(np.max(X)+1)[X]