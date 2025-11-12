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
import matplotlib.pyplot as plt

# Create classifier
class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Fit model (optimize regression curve through gradient descent)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        weights_arr = np.zeros(self.n_iters)
        bias_arr = np.zeros(self.n_iters)

        # Gradient descent
        for i in range(self.n_iters):
            """
            Perform n_iters iterations of gradient descent:
                1. Generate predictions for y values for each X value. The
                inputs and predictions have a linear relationship because each
                input is multiplied by a constant weight and added to a
                constant bias. 
                2. Get the derivatives of weight by multiplying the dot product
                of each input value by the difference between the true and
                predicted output values by 1/number of samples. Any number
                above the regression line will "push" the derivative of weight
                towards negative since the slope needs to be more positive to
                get the regression line closer to those data points, minimizing
                the cost function. The opposite is true for points whose true
                output is lower than their predicted output. The dot product of
                these terms the sum of each of these terms. The sign of the dot
                product determines the sign of the derivative, which informs
                which direction gradient descent is performed. A negative
                derivative pushes the weight into the positive direction and
                vice versa.
                3. Get the derivative of bias by summing the differences
                between the true and predicted output and multiplying it by the
                sample scaler 1/number of samples. Since the value of the input
                is not relevant for the value of the bias, inputs are ignored.
                A negative derivative of bias pushes the bias into the positive
                and vice versa.
            """
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            weights_arr[i] = self.weights
            bias_arr[i] = self.bias

        fig, axs = plt.subplots(2, 1)
        functions = [weights_arr, bias_arr]
        titles = ['Weight (Slope)', 'Bias (Intercept)']
        for i, ax in enumerate(axs):
            ax.plot(functions[i])
            ax.set_ylabel(titles[i])
            ax.set_xlabel('Iterations')
        fig.suptitle("Gradient descent of regression components over iterations")
        plt.tight_layout()
        plt.show()

        print(self.weights)
        print(self.bias)

    # New test samples used to predict and return value
    def predict(self, X):
        """
        Predict the output for each input using the weight and bias calculated
        by applying the linear regression model to each input. Each predicted
        output will be compared to the corresponding true output and the MSE
        will be calculated in the function mse in linregtest.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
