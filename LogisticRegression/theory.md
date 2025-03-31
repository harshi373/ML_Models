# Logistic Regression: Understanding and Internal Working

## 1. Introduction

Logistic Regression is a supervised learning algorithm used for **binary classification problems**. It predicts the probability that a given input belongs to a particular class. Unlike Linear Regression, which predicts continuous values, Logistic Regression maps inputs to probabilities using the **sigmoid function**.

## 2. Why Not Linear Regression for Classification?

Linear Regression is not suitable for classification because:

- It can predict values outside the range [0,1], which are invalid probabilities.
- It does not provide a clear decision boundary for classification.
- It is highly sensitive to outliers, which can distort predictions.
- It assumes a linear relationship between input features and output, which is often not the case for classification.

## 3. Logistic Regression Hypothesis Function

Instead of using a linear function, Logistic Regression applies the **sigmoid function**:

$$
sigma(z) = \frac{1}{1 + e^{-z}}
$$

where:
\(z = W^T X + b\)

- **X**: Input features
- **W**: Weights (coefficients)
- **b**: Bias (intercept)

The output of this function is always between **0 and 1**, making it suitable for probability estimation.

## 4. Cost Function: Binary Cross-Entropy Loss

Logistic Regression uses the **Binary Cross-Entropy Loss (Log Loss)** instead of Mean Squared Error (MSE) because MSE leads to non-convex loss functions for classification problems. The loss function is:

$$
J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \right]
$$

where:

- **y\_i**: Actual label (0 or 1)
- **\hat{y\_i}**: Predicted probability
- **m**: Number of samples

This function penalizes incorrect predictions more heavily, ensuring better model performance.

## 5. Optimization: Gradient Descent

To minimize the loss function, we update the weights and bias using **Gradient Descent**:

$$
W = W - \alpha \frac{\partial J}{\partial W}
$$

$$
b = b - \alpha \frac{\partial J}{\partial b}
$$

where:

- **\alpha**: Learning rate
- **\frac{\partial J}{\partial W}**: Gradient of the cost function w\.r.t. weights
- **\frac{\partial J}{\partial b}**: Gradient of the cost function w\.r.t. bias

The gradients are computed as:

$$
\frac{\partial J}{\partial W} = \frac{1}{m} \sum (h(X) - y) X
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum (h(X) - y)
$$

## 6. Decision Boundary

Since Logistic Regression outputs probabilities, we classify a sample as **1** if:

$$
\sigma(W^T X + b) \geq 0.5
$$

and **0** otherwise. This threshold (0.5) can be adjusted based on application needs.

## 7. Summary

- Logistic Regression is used for **binary classification**.
- It applies the **sigmoid function** to map inputs to probabilities.
- It optimizes the model using **binary cross-entropy loss** and **gradient descent**.
- It provides a clear **decision boundary** for classification tasks.

**Logistic Regression can be extended to multiclass classification**, but the standard form is only for **binary classification**. To handle **multiple classes (more than two)**, we use two common strategies:

---

### **1. One-vs-Rest (OvR) / One-vs-All (OvA)**
- This approach trains **multiple binary logistic regression models**, one for each class.
- For **K classes**, we train **K binary classifiers**, each distinguishing **one class vs. all others**.
- The final class is chosen based on the highest probability.

🔹 **Example**: Suppose we have **3 classes**: `A, B, C`  
- Train 3 models:
  - Model 1: Classify A vs. (B, C)
  - Model 2: Classify B vs. (A, C)
  - Model 3: Classify C vs. (A, B)
- For a new input, the model with the **highest probability wins**.

**Pros**: Simple, works well with logistic regression  
**Cons**: Independent classifiers may lead to inconsistencies  

---

### **2. Softmax Regression (Multinomial Logistic Regression)**
- Instead of **multiple binary classifiers**, this method **generalizes logistic regression** to directly classify into multiple classes.
- Uses the **softmax function** to compute probabilities for each class:

\[
P(y = k | X) = \frac{e^{W_k^T X + b_k}}{\sum_{j=1}^{K} e^{W_j^T X + b_j}}
\]

where:
- \( K \) is the number of classes
- \( W_k \) and \( b_k \) are the weight vector and bias for class \( k \)
- The denominator ensures probabilities sum to **1**.

🔹 **Example**: If we have **3 classes (A, B, C)** and an input **X**,  
Softmax will return probabilities:  
- P(A) = 0.2  
- P(B) = 0.5  
- P(C) = 0.3  

We assign the input to **Class B** since it has the highest probability.

 **Pros**: More consistent than OvR, works well for multiclass classification  
 **Cons**: Computationally expensive for many classes  

---

### **Which One to Use?**
- **One-vs-Rest (OvR)**: Works well for smaller datasets, easier to implement.
- **Softmax Regression**: Preferred for deep learning and large datasets.

