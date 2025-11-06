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


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        # Defined later
        self.weights = None
        self.bias = None

    # Involves training and gradient descent
    def fit(self, X, y):
        # In the case of 2D
        # init parameters (gradient descent needs to start somewhere
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # for each component, put a 0
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            # Multiply each weight component by feature vector component, one value for each sample
            y_predicted = np.dot(X, self.weights) + self.bias  # np.dot = multiplication
            # Derivative of weight, one value for each feature vector component
            # X.T: along other axis (transpose)
            dw = (1 / n_samples) * np.dot(
                X.T, (y_predicted - y)
            )  # Multiply each sample with predicted value, and sum it up
            # Derivative of bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # New test samples used to predict and return value
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
