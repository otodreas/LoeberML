"""
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
def euclidian_distance(x1, x2):  # Euclid. dist. between 2 feature vectors
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Create class KNN
class KNN:

    # Take k (number of nearest neighbors to consider)
    def __init__(self, k=3):
        self.k = k

    # Fit training samples and training labels (training step)
    def fit(self, X, y):
        """
        This method is only here by scikitlearn convention. This
        algorithm does not require a training step. Simply return
        X and y as X_train and y_train."""
        self.X_train = X
        self.y_train = y

    # Predict new samples
    def predict(self, X):
        # This method gets multiple samples
        predicted_labels = [
            self._predict(x) for x in X
        ]  # Call helper _predict method on each x in X
        return np.array(predicted_labels)

    # This helper method only gets one sample (lowercase)
    def _predict(self, x):
        # Calculate all euclidean distances to KNNs
        # Compute distances of x to all training samples
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[: self.k]
        # Get labels of nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Using majority vote, get most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
