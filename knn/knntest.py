# Import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn import KNN


# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = KNN(k=3)  # Create classifier
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)  # how many of our predictions are correctly classified?
print(acc)