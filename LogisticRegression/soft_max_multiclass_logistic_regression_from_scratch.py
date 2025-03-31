#logistic regression for multi class classification
import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Avoid overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)   # Normalize
    
    def one_hot_encode(self, y, num_classes):
        m = y.shape[0]
        y_one_hot = np.zeros((m, num_classes))
        y_one_hot[np.arange(m), y] = 1
        return y_one_hot

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Add small value for numerical stability

    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))

        # Initialize weights and bias
        self.weights = np.random.randn(n, num_classes)
        self.bias = np.zeros((1, num_classes))

        # Convert labels to one-hot encoding
        y_one_hot = self.one_hot_encode(y, num_classes)

        for epoch in range(self.epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(logits)

            # Compute loss
            loss = self.compute_loss(y_one_hot, y_pred)

            # Backpropagation
            grad_w = (1/m) * np.dot(X.T, (y_pred - y_one_hot))
            grad_b = (1/m) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)

            # Update weights
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(logits)
        return np.argmax(y_pred, axis=1)  # Choose class with highest probability

# Generate dummy dataset (4 samples, 3 features, 3 classes)
X_train = np.array([
    [1.2, 0.7, 2.1],
    [0.5, 1.5, 1.7],
    [2.1, 1.9, 0.3],
    [1.7, 2.2, 1.0]
])
y_train = np.array([0, 1, 2, 1])  # 3 classes (0, 1, 2)

# Train Softmax Regression
model = SoftmaxRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict on new data
X_test = np.array([[1.5, 1.8, 1.2]])
prediction = model.predict(X_test)
print("Predicted Class:", prediction)
