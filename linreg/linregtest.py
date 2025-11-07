import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linreg import LinearRegression

# Make synthetic dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
# In this case, y data are represented on the y axis of the plot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Assign linear regression class to regressor
regressor = LinearRegression(lr=0.1, n_iters=100)
# Perform gradient descent
w, b = regressor.fit(X_train, y_train)

fig, ax = plt.subplots(3, 1)
ax[0].plot(w)
ax[1].plot(b)
ax[0].set_title("Weight")
ax[1].set_title("Bias")

# Test the model
predicted = regressor.predict(X_test)


# Generate mean squared error
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


# Print performance
mse_value = mse(y_test, predicted)
print(f"Mean squared error: {round(mse_value, 1)}")


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
m1 = ax[2].scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = ax[2].scatter(X_test, y_test, color=cmap(0.5), s=10)
ax[2].plot(X, y_pred_line, color="k", linewidth=2, label="prediction")
ax[2].set_title("Regression")
plt.tight_layout()
plt.show()
