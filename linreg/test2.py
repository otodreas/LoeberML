import numpy as np
from matplotlib import pyplot as plt
from linreg import LinearRegression

X, y = np.array([[0], [1], [2]]), np.array([-1, 0, 1])

regressor = LinearRegression(lr=.5, n_iters=50)
regressor.fit(X, y)  # generate model

# Generate plot
fig2, ax2 = plt.subplots()
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
m1 = ax2.scatter(X, y, color=cmap(0.9), s=10)
m2 = ax2.scatter(X, y, color=cmap(0.5), s=10)
ax2.plot(X, y_pred_line, color="k", linewidth=2, label="prediction")
ax2.set_title("Regression")
plt.tight_layout()
plt.show()