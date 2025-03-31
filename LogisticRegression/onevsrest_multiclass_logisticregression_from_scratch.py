import numpy as np

class OneVsRestLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.models = {}  # Store models for each class

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        m = len(y)
        return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))  # Cross-entropy loss

    def train_binary_logistic_regression(self, X, y):
        m, n = X.shape
        weights = np.random.randn(n)
        bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, weights) + bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            grad_w = (1/m) * np.dot(X.T, (y_pred - y))
            grad_b = (1/m) * np.sum(y_pred - y)

            # Update weights and bias
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return weights, bias

    def fit(self, X, y):
        self.classes = np.unique(y)  # Get unique class labels
        for c in self.classes:
            print(f"Training classifier for class {c} vs rest...")
            binary_y = (y == c).astype(int)  # Convert class labels to binary (1 for class c, 0 for others)
            weights, bias = self.train_binary_logistic_regression(X, binary_y)
            self.models[c] = (weights, bias)

    def predict(self, X):
        predictions = []
        for c, (weights, bias) in self.models.items():
            z = np.dot(X, weights) + bias
            prob = self.sigmoid(z)
            predictions.append(prob)

        predictions = np.array(predictions).T  # Shape (num_samples, num_classes)
        return np.argmax(predictions, axis=1)  # Pick class with highest probability

# Example dataset (5 samples, 3 features, 3 classes)
X_train = np.array([
    [2.5, 1.7, 3.1],
    [1.1, 2.0, 2.5],
    [3.2, 3.1, 1.1],
    [2.8, 2.5, 1.5],
    [1.5, 2.7, 3.0]
])
y_train = np.array([0, 1, 2, 2, 1])  # Classes: 0, 1, 2

# Train the One-vs-Rest Logistic Regression Model
model = OneVsRestLogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict on a new sample
X_test = np.array([[2.0, 2.1, 2.5]])
prediction = model.predict(X_test)
print("Predicted Class:", prediction)
