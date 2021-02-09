import numpy as np 


class Perceptron:
	def __init__(self, X, y):
		'''
		Parameters:
		------------
		X: {matrix/vector} A (m x n) feature matrix, which rows represent each training example and columns represent each feature. If it is a vector (data has only 1 feature), then the shape must be (m x 1).
		Y: {vector} A (m x 1) label matrix, which rows represent each training example. This vecotr is the true label vector, which is used to fit the paramters to an equation. 
		'''
		self.X = np.c_[np.ones(X.shape[0]), X]
		self.y = y.reshape(-1,1)

	def fit(self, a=0.01, n=1500):
		m = self.X.shape[0]
		n_ = self.X.shape[1]

		self.W = np.zeros((1,n_))
		self.costs = []
		for i in range(n):
			predictions = np.array(self.predict_class()) # (mx1 vector) # 99, 1
			errors = np.subtract(predictions, self.y)
			update = a * np.dot(errors.T, self.X) # (n,1) vector
			self.W = self.W - update # (1,n) vector


	def predict(self, X):
		'''
		Returns:
		-----------
		The net output of the given data, Z= X*WT, which is in (mx1) column vector shape.

		Formula:
		---------
		z = W0X0 + W1X1 + W2X2 ... + WnXn
		which is:
		z = X * WT (X=(mxn) matrix, W=(1xn) column vector)
		
		Note that W0 is the bias-term, and X0 is 1 for the convenience
		'''
		return np.dot(X, self.W.T) # (mx1) vector

	def predict_class(self):
		'''
		Returns an array containing all the classifications.
		'''
		predictions = self.predict(self.X)
		preds = []
		for pred in predictions:
			preds.append(np.where(pred > 0, 1, -1))
		return preds



