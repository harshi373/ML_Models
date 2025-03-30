#single linear regression
import numpy as np
X=np.array([1,2,3,4,5])
y=np.array([2,4,5,4,5])
w=0.0
b=0.0
#y=wx+b
learning_rate=0.01
epochs=1000
n=len(X)
#loss function sum((actual-predicted)^2)/n
#gradient descent algorithm
for _ in range(epochs):
  y_pred=w*X+b
  dw=(-2/n)*sum(X*(y-y_pred)) #gradient(dervative) of loss function wrt w 
  db=(-2/n)*sum(y-y_pred) #gradient(dervative) of loss function wrt b
  w-=learning_rate*dw
  b-=learning_rate*db

print(f"final weight (w):{w}")
print(f"final bias (b):{b}")

y_pred=w*X+b
print(f"predicted values:{y_pred}")