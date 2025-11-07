"""
Linear regression: fitting a straight line to data.
Predict continuous data, not class.
The line is found with a cost function
The cost function for a linear regression is MSE (mean squared error)
We look for the minimum of the MSE
To find the minimum, we need the derivitave of MSE
We use gradient decent to find the minimum MSE (iterative process)
    - Weight and bias are initialized
    - Go in direction of steepest descent (in the negative direction of the
    gradient)
    - Each iteration updates weight and bias based on some rule
        New weight = previous weight - learning rate * derivative (gradient slope)
            Learning weight defines how far we step at each iteration
            Speed-precision tradeoff
"""

import numpy as np

# Create classifier
class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        # Assign variables
        self.lr = lr
        self.n_iters = n_iters
        # Define weights and bias later
        self.weights = None
        self.bias = None

    # Fit model (optimize regression curve through gradient descent)
    def fit(self, X, y):
        # Initialize parameters (gradient descent needs to start somewhere)
        # Get number of samples and number of features
        """
        The fit function fits a regression model to the data using the
        following procedure:
            1. Initialize weights and bias at 0
                a. One weight per feature (dimension of X)
                b. One bias for the whole model
            2. Initialize a loop for the number of iterations set by the user
                a. Predict y value for each data point in each iteration
                b. Get the derivative of the weights and bias at each iteration
                    i.  Derivative of weight: 1/n * X*(y_pred-y). As guesses
                    get better, y_pred-y approaches 0, entire term approaches
                    0, weight stops improving.
                    ii. Derivative of bias: 1/n * sum(y-y_pred). Same logic for
                    y-y_pred applies.
                c. Update weights and biases. derivative of weight OR bias *
                learning rate is SUBTRACTED from weight or bias. Learning rate
                determines how large a gradient descent step is for each
                iteration
        """
        n_samples, n_features = X.shape
        # One weight per feature, bias is 0 for the whole model
        self.weights = np.zeros(n_features)  # for each component, put a 0
        self.bias = 0

        # Gradient descent
        dws = np.zeros(self.n_iters)
        dbs = np.zeros(self.n_iters)
        for i in range(self.n_iters):
            # Multiply each weight component by feature vector component, one value for each sample
            y_predicted = np.dot(X, self.weights) + self.bias  # np.dot = multiplication
            # if i == 0 or i == self.n_iters - 1:
            #     print(f"X: i={i}, {X[0][:10]}")
            #     print(f"y_pred: i={i}, {y_predicted[:10]}")
            # Derivative of weight, one value for each feature vector component
            dw = (1 / n_samples) * np.dot(
                X.T, (y_predicted - y)
            )  # Multiply each sample with predicted value, and sum it up
            # Derivative of bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            if i == 0 or i == self.n_iters - 1:
                print(f"1/N*X*y_pred-y: {X.T.shape}, {(y_predicted - y).shape}")

            # dws[i] = dw
            # dbs[i] = db



            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db



        # return dws, dbs

    # New test samples used to predict and return value
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
