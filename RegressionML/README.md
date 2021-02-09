## Introduction

This is a simple **univariate** / **multivariate** linear regression project for my Mathematics project. 

This project is written in Python 3.7 and with packages NumPy, Matplotlib, Pandas, and Sklearn

I used NumPy to:
- Perform the calculations in a vectorized way to be more efficient
- It is also an extended version of the current Math library of Python

I used Matplotlib to:
- Plot the graphs in both 2D and 3D

I used Sklearn to:
- Split the data into 2 parts: Training and Testing sets
- To access the datasets of the library

I used Pandas to:
- Getting the data from a .csv file
- Easy to use with NumPy

Math included in this project:
- Statistics (MSE, Gradient Descent, Regression)
- A bit of Calculus (Gradient Descent)
- Linear Algebra (Making the algorithm in a vectorized way - Using Matrices and Vectors in calculations)

## Installations

To use this code, first of all you need to install python3. You can download python3 from their official website. (https://www.python.org/downloads/)

To install the libraries the code depends on, you need to run:
`pip install -r requirements.txt` in your shell, which will install all the packages this environment uses with their correct versions.

## Usage

This code comes with an example .csv file to test the code yourself. If you want to use another data, you can simply type the location of the file into the `pd.read_csv(YOUR_URL)` part. 

Classes:
- Graph (in grapher.py)
- LinearRegression (in regression.py)

In order to use the `Graph` class, you need to import it using `from RegressionML import Graph`. 
In order to use the `LinearRegression` class, you need to (again) import it using `from RegressionML import LinearRegression`

### Grapher
This class is used to plot the data in both 2D and 3D with Matplotlib. 
There are 4 methods in this class:
- graph_2D(): Graphs a 2D equation into a cartesian plane
- scatter_2DData(): Scatters the data points into a cartesian plane
- graph_3D(): Graphs a 3D equation into a 3D space
- scatter_3DData(): Scatters 3D points into a 3D space

### LinearRegression
This class is used for fitting the data into an equation and predicting further data.
There are 3 methods in this class:
- `fit()`: Fits an equation according to given X, Y data
- `generate_dataset()`: Generates a new dataset based on the given parameters using the sklearn's dataset generation library
- `sk_split()`: Splits the data into training and testing examples using sklearn's train_test_split() method
- `feature_scale()`: Applies the feature scaling technique to input parameters
- `mean_normalize()`: Returns a mean-normalized data, which has approx. 0 mean and unit variance
- `mse()`: Returns a MSE value based on the parameters
- `normal()`: Fits the parameters into the data, without any iterations. Use this method if you have examples less than 500, which is quicker than the `fit()` method.
- `generate_equation()`: Returns an **ARRAY**, based on the input value, x. Input value x is the (m x n) example-feature matrix. Use for plotting a 2D equation.

Also, you can have an idea of the code from the comments and docstrings I wrote. 
If you find any bug or want to add any feature to this project, please email me: emirhurturk444@gmail.com

Thanks for being interested in,