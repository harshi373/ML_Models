#logistic regression
import numpy as np
class LogisticRegression:
  def __init__(self,learning_rate=0.01,epochs=1000):
    self.learning_rate=learning_rate
    self.epochs=epochs
    self.weights=None
    self.bias=None
  def sigmoid(self,z):
    return 1/(1+np.exp(-z))
  def fit(self,X,y):
    m,n=X.shape
    self.weights=np.zeros(n)
    self.bias=0
    for _ in range(self.epochs):
      linear_model=np.dot(X,self.weights)+self.bias
      predictions=self.sigmoid(linear_model)
      dw=(1/m)*np.dot(X.T,(predictions-y))
      db=(1/m)*np.sum(predictions-y)
      self.weights-=self.learning_rate*dw
      self.bias-=self.learning_rate*db
  def predict(self,X):
      linear_model=np.dot(X,self.weights)+self.bias
      y_pred=self.sigmoid(linear_model)
      return (y_pred >= 0.5).astype(int)
if __name__ == "__main__":
    # Sample dataset (X: Features, y: Labels)
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Predictions:", predictions)