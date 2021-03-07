import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from RegressionML.logger import Logger

class LinearRegression:
    def __init__(self):
        self.__W = None
        self.__logger = Logger()

    def normal(self, X, y):
        """
        :param X: (numpy array) (shape: m x n) The feature-example matrix
        :param y: the target vector (numpy array) shape: (m, 1)
        :return: Self
        """

        self.__W = np.linalg.pinv(np.dot(np.dot(np.dot(X.T, X), X.T), y))

        return self

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
        self.__W = np.zeros(n)
        self.cost_history_ = []
        self.rsq_history = []
        # print(self.predict(X).shape) # (4,)
        iter_number = 0
        for _ in range(n_iters):
            try:
                if np.abs(self.cost_history_[-1] - self.cost_history_[-2]) < 0.001:
                    print(f"Change in cost is less than 0.001 in iteration {iter_number}, exitting iterations...")
                    break
            except IndexError:
                pass
            finally:
                predictions = self.predict(X)
                diff = predictions - y
                update = a * (2/m) * np.dot(diff.T, X)
                self.__W = self.__W - update
                cost = self.mse(y, predictions)
                rsqerr = self.rsq(predictions, y)
                self.cost_history_.append(cost)
                self.rsq_history.append(rsqerr)
                iter_number+=1

                if (iter_number % 100 == 0):
                    self.__logger.msg(list(self.__W), iter_number, cost, rsqerr)

        self.__logger.log(iter_number, self.cost_history_[-1], self.rsq_history[-1])
        return self

    def mse(self, y, y_hat):
        return np.mean((y_hat - y)**2)

    def predict(self, X):
        """
        Returns the predicted output according to the input parameter X.

        :param X: (numpy array) (shape: m x n) The feature-example matrix
        :return y: (numpy array) (shape m x 1) The predictions based on the weights
        """
        return np.dot(X, self.__W.T)

    @staticmethod
    def feature_scale(X):
        """
        :param X: Data that is going to be feature-scaled
        :return: Feature Scaled Data
        """
        return X / np.std(X)

    @staticmethod
    def mean_normalize(X):
        """
        :param X: Data that is going to be mean normalized
        :return: Mean normalized data, which has a mean of approx. 0 and a unit variance
        """
        return (X - np.mean(X))/np.std(X)

    @staticmethod
    def train_test_split(X, y, train_percent=0.8, seed=2):
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

    @staticmethod
    def generate_dataset(samples, features, seed):
        X, y = make_regression(n_samples=samples, n_features=features, noise=20, random_state=seed)
        return X, y

    @staticmethod
    def sk_split(X, y, train_percentage = 0.8):
        return train_test_split(X, y, test_size = 1-train_percentage, random_state=2)

    @property
    def W(self):
        return self.__W

    def rsq(self, preds, y):
        rss = np.sum((y-preds)**2)
        tss = np.sum((y-np.mean(y)) ** 2)
        rsq = 1 - (rss/tss)
        return rsq


