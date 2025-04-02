#KNN classification
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data (lazy learning)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """Predict class labels for test data"""
        predictions = []
        for x_test in X_test:
            distances = [self._euclidean_distance(x_test, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

# Example Usage
X_train = np.array([[1.2, 3.1], [2.3, 3.3], [3.1, 2.2], [5.5, 1.8], [6.1, 2.0]])
y_train = np.array(['A', 'A', 'B', 'B', 'B'])

X_test = np.array([[2.5, 3.0], [5.8, 2.1]])

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print("Predicted classes:", predictions)

