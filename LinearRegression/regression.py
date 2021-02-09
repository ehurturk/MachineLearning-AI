import numpy as np


class LinearRegression:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.W = None
        self.b = None
        self.costs = []

    def fit(self, a, n):
        '''
        Parameters:
        --------------
        X: {array} A 2D m x n matrix. The first column of this must be 1 for all rows, as the feature x0 w
    would equal 1, for the bias convention. This matrix is a example x feature matrix, which rows represent each training example and columns represent each feature.
        Y: {array} A m x 1 vector. This vector represents the true labels for each training example.
        W: {array} Th e weight matrix which must be 1 x X.shape[1].
        a: {float} The learning rate which is responsible for controlling the gradient descent algorithm.
        n: {int} The number of iterations

        Returns:
        --------------
        W: {array} An array containing weights of the each input feature.

        Formula:
        --------------
        for n iterations:
            W = W - (a * (1/X.shape[0]) * (X.dot(W.T)-real).dot(X)
        '''
        self.W = np.zeros(self.X.shape[1])
        self.b = 0
        
        
        for i in range(n):
            predictions = np.dot(self.X, self.W) + self.b
            deltaW = (a / self.X.shape[0]) * np.dot(self.X.T,(predictions - self.Y)) # for some reason shape deltaW is 2,80
            deltaB = (a / self.X.shape[0]) * np.sum(predictions - self.Y)
            self.W = self.W - deltaW
            self.b = self.b - deltaB
            self.costs.append(self.mse(self.Y, predictions))
        return self

    def mse(self, true, pred):
        return np.mean((true - pred) ** 2)

    def predict(self, test_data):
        return np.dot(test_data, self.W) + self.b

    def graph(self, xy):
        result = 0
        for i in range(self.W.shape[0]):
            result = result + self.W[i] * xy[i]
        return result + self.b


