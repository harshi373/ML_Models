Linear Regression is one of the simplest and most widely used supervised learning algorithms for regression tasks. It is used to establish a relationship between a dependent variable (Y) and one or more independent variables (X). The relationship is modeled using a straight line.
there are two types of linear regression 
single linear regression (one independent variable and one dependent vaeiable) y=mx+c
multiple linear regression(more than one independent variable and one dependent variable)y=w1x1+w2x2+.....+ Wnxn+b
in above equation m,c,w1,w2,...,b are parameters 
our task is to find find best fit by adjusting these parameters in such a way that it has to minimise loss function (mean squared difference between actual value and predicted value)
adjusting of paramters is done by gradient descent where each parameter is adjusted like w1=w1-(learning_parameter*derivative of loss function)
loss_function=1/nsum((predicted value -actual values)^2)