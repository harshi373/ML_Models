#multiple linear regression
import numpy as np
X=np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])
y=np.array([3,6,7,8,11])
m,n_features=X.shape
W=np.zeros(n_features)
b=0.0
learning_rate=0.01
epochs=1000
#y=w1x1+w2x2+b
#gradient descent algorithm
for _ in range(epochs):
  y_pred=np.dot(X,W)+b
  dw=(-2/m)*np.dot(X.T,(y-y_pred))
  db=(-2/m)*np.sum(y-y_pred)
  w-=learning_rate*dw
  b-=learning_rate*db
print(f"final weight (w):{W}")
print(f"final bias (b):{b}")
y_pred=np.dot(X,W)+b
print(f"predicted values:{y_pred}")