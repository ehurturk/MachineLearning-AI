import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self):
        self.W = None

    def normal(self, X, y):
        """
        :param X: (numpy array) (shape: m x n) The feature-example matrix
        :param y: the target vector (numpy array) shape: (m, 1)
        :return:
        """
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    def fit(self, X, y, a=0.01, n_iters=1000):
        """
        :param X: (numpy array) (shape: m x n) The feature-example matrix
        :param y: the target vector (numpy array) shape: (m, 1)
        :param a: The learning rate (default = 0.01)
        :param n_iters: Number of iterations to fit the model (default = 1000)
        :return self
        """
        X = np.c_[np.ones(X.shape[0]), X]
        m, n = X.shape
        self.W = np.zeros(n)
        self.cost_history_ = []
        #print(self.predict(X).shape) # (4,)
        iter_number = 0
        for _ in range(n_iters):
            try:
                if (np.abs(self.cost_history_[-1] - self.cost_history_[-2]) < 0.001):
                    break
            except IndexError:
                pass
            finally:
                predictions = self.predict(X)
                diff = predictions - y
                update = a * np.dot(diff.T, X)
                #print(update.shape) # (2,)
                #print(self.W.shape) # (2,)
                self.W = self.W - update
                self.cost_history_.append(self.mse(y, predictions))
                iter_number+=1
        return self

    def mse(self, y, y_hat):
        return np.mean((y_hat - y)**2)

    def predict(self, X):
        """
        Returns the predicted output according to the input parameter X.

        :param X: (numpy array) (shape: m x n) The feature-example matrix
        :return y: (numpy array) (shape m x 1) The predictions based on the weights
        """
        return np.dot(X, self.W.T)

    def feature_scale(self, X):
        """
        :param X: Data that is going to be feature-scaled
        :return: Feature Scaled Data
        """
        return X / np.std(X)


    def mean_normalize(self, X):
        """
        :param X: Data that is going to be mean normalized
        :return: Mean normalized data, which has a mean of approx. 0 and a unit variance
        """
        return (X - np.mean(X))/np.std(X)

    def train_test_split(self, X, y, train_percent=0.8, seed=2):
        """

        :param X: The (m x n) example-feature matrix
        :param y: The target vector (m x 1)
        :param train_percent: The train-test-split percentage (Default=0.8)
        :param seed: The seed for the random shuffling (Default=None)
        :return: X training data (can be used for fitting), X testing data (can be used for testing), y training data (can be used for fitting), y testing data (can be used for testing)
        """
        if (seed != None):
            np.random.seed(seed)
        np.random.shuffle(X)
        np.random.shuffle(y)
        m = X.shape[0]
        div_val1 = int(m*train_percent) # 3
        div_val2 = int(np.ceil((m*(1-train_percent)))) # 1

        return X[:div_val1], X[div_val1: div_val1+div_val2+1], y[:div_val1], y[div_val1: div_val1+ div_val2+1]

    def generate_equation(self, X):
        """

        :param X: The (m x n) example-feature matrix
        :return: X_ (a linspace min: min value of X, max: max value of X), predictions (predictions based on that linspace)
        """
        X_ = X
        X__ = np.c_[np.ones(X_.shape[0]), X_]
        predictions = self.predict(X__)
        return X_, predictions

    def generate_dataset(self, samples, features, seed):
        X, y = make_regression(n_samples=samples, n_features=features, noise=20, random_state=seed)
        return X, y

    def sk_split(self, X, y, train_percentage = 0.8):
        return train_test_split(X, y, test_size = 1-train_percentage, random_state=2)


# TODO:
# Implement RMSE, and R2
# Implement a logging system in which program logs the output of the data