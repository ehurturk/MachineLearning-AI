from RegressionML import LinearRegression
from RegressionML import Graph

reg = LinearRegression()
graph = Graph()

X_, y = LinearRegression.generate_dataset(samples=500, features=1, seed=2) # Generating a dataset with 100 samples with only 1 feature with a random seed 1
X = LinearRegression.mean_normalize(X_) # Scaling the features the make gradient descent converge faster
X_train, X_test, y_train, y_test = LinearRegression.sk_split(X_, y, 0.8) # Splitting the original data to train and test the model
reg.fit(X_train, y_train, 0.001, 10000) # Train the model using the training examples

print(f'Parameters:\n----------------') # Log the parameters
for i in range(reg.W.shape[0]):
    print(f'{i}) {reg.W[i]}')

input_x, output_y = reg.generate_equation(X_) # Generate equation to plot
graph.scatter_2Ddata(X_train, y_train, X_test, y_test) # Plot the dataset
graph.graph_2D(input_x, output_y, 'Dataset', 'Feature', 'Output') # Plot the equation
graph.graph_2D(range(len(reg.cost_history_)), reg.cost_history_, 'Graph of Convergence of Cost Function', 'Iterations', 'Cost Function') # Plot the convergence of the cost function



