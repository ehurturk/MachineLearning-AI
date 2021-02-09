import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from regression import LinearRegression
from grapher import Grapher


# Get the Data
X, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Inits
grapher = Grapher()
regression = LinearRegression(X_train, y_train)


# Fit the parameters
regression.fit(0.01, 1000)
predicted_labels = regression.predict(X)


# Graph 3D Points of the Data
grapher.graph3Dscat(X[:, 0], X[:, 1], y) # grapher.graph3dscat(first_feature_vector, second_feature_vector, label_vector)


# Graph 3D Regression Equation
x1 = np.linspace(-6, 6, 30)
x2 = np.linspace(-6, 6, 30)
X1, X2 = np.meshgrid(x1, x2)
Y = regression.graph([X1, X2])
print(Y.shape)
grapher.graph3Deq(X1, X2, Y)


# Graph Cost Function
grapher.graph2Deq(range(1, len(regression.costs) + 1), regression.costs, 'Graph of Convergence of the Cost Function', 'Number of Iterations', 'MSE Error Value')

