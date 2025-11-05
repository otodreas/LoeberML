# Import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from knn import KNN


# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target  # X: values. y: classes
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create classifier
clf = KNN(k=3)
# Fit data (in this case, this step is only necessary for assigning variables)
clf.fit(X_train, y_train)
# Generate predictions using the entire X_test set
predictions = clf.predict(X_test)

# Check how many of our predictions are correctly classified
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

# Plot
fig, ax = plt.subplots()
ax.scatter(X_test[:, 2], X_test[:, 3], c=y_test)
fig.suptitle('knn predictions (color = predicted class)')
plt.show()