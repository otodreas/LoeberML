"""
This file tests the algorithms developed in knn.py on the iris dataset.
"""

# Import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from knn import KNN


# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target  # X: values. y: classes
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create classifier where k=3
clf = KNN(k=3)
# Fit data (in this case, this step is only necessary for assigning variables)
clf.fit(X_train, y_train)
# Generate predictions using the entire X_test set
predictions = clf.predict(X_test)

# Check how many of our predictions are correctly classified
# Correct classifications are stored in y_test
acc = np.sum(predictions == y_test) / len(y_test)

# Print summary
print(f"True class:      {y_test}")
print(f"Predicted class: {predictions}")
print(f"Accuracy of classifier: {round(acc * 100, 2)}%")

# Create figure, plot distribution of each feature by class
fig, axs = plt.subplots(X_train.shape[1], 1)
for i, col in enumerate(range(X_train.shape[1])):
    dist = X_train[:, col]
    legend = True if i == 0 else False
    sns.boxplot(x=dist, hue=y_train, legend=legend, ax=axs[i])

axs[0].legend(bbox_to_anchor=(1, 1), loc=2)
plt.suptitle("Distributions of features by class")
plt.tight_layout()
plt.show()
