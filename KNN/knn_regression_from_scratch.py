import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store the training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        """Predict the target for each sample in X."""
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict the target for a single sample."""
        distances = np.linalg.norm(self.X_train - x, axis=1)
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_targets = self.y_train[nearest_indices]
        return np.mean(nearest_targets)

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [1.5, 2.0, 2.5, 3.5, 5.0]
    
    X_test = [[1.5], [3.5], [4.5]]
    
    knn = KNNRegressor(k=2)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
