import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linreg import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression(lr=.1, n_iters=100)
regressor.fit(X_train, y_train)  # generate model
predicted = regressor.predict(X_test)  # test model

# Generate mean squared error
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

# Print performance
mse_value = mse(y_test, predicted)
print(f"Mean squared error: {round(mse_value, 1)}")

# Generate plot
fig, ax = plt.subplots()
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
m1 = ax.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = ax.scatter(X_test, y_test, color=cmap(0.5), s=10)
ax.plot(X, y_pred_line, color="k", linewidth=2, label="prediction")
ax.set_title("Regression")
plt.tight_layout()
plt.show()
