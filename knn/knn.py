"""
This file defines the functions used to classify with knn

KNN DEMO Notes
    - KNN: sampleS classified by DNN
    - Each data point is represented by a feature
    vector (x, y coordinates)
    Method:
        1. Aquire labeled training data
        2. For each new point, calculate distance to
        each other nearest neighbor
        3. Predict label of the new point based on
        most common class in its vascinity
        4. Distances calculated with euclidean
        distance squrt((x1-x2)^2+(y1-y2)^2) in 2D
            - Just add dimensions for more high
            dimensional data

"""

# Import libraries
import numpy as np
from collections import Counter


# Calculate euclidian distance between two points in one dimension
def euclidian_distance(x1, x2):
    """
    This function will be applied in a loop, one iteration per dimension.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Create class KNN
class KNN:
    # Take k (number of nearest neighbors to consider)
    def __init__(self, k=3):
        self.k = k

    # Fit training samples and training labels (training step)
    def fit(self, X, y):
        """
        KNN does not require a training step. We still maintain this method to
        access the training set if we need it.
        """
        self.X_train = X
        self.y_train = y

    # Predict class of all points in the testing set
    def predict(self, X):
        """
        This method takes help from the class _predict() to predict the class
        of every data point in the testing set, represented by capital X.
        """
        # Call helper _predict method on each x in X
        predicted_labels = [self._predict(x) for x in X]
        # Return all predicted labels
        return np.array(predicted_labels)

    # Predict class of a given data point in the testing set
    def _predict(self, x):
        """
        This helper method predicts the class of test data point x by getting
        the euclidean distance to each data point in the training set and
        returning the class of the majority of the k nearest neighbors.
        """
        # Compute distances of x to all training samples
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[: self.k]
        # Get labels of nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Using majority vote, get most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        # Return only the most common label for the k nearest neighbors
        return most_common[0][0]
