# K-Nearest Neighbors (KNN) Algorithm

## Introduction
K-Nearest Neighbors (KNN) is a **non-parametric, instance-based** machine learning algorithm used for **classification and regression**. It makes predictions based on the similarity between the input sample and its nearest neighbors using a distance metric such as **Euclidean distance, Manhattan distance, or Minkowski distance**.
## How KNN Works
KNN is a **lazy learning algorithm**, meaning it does not explicitly learn a model during the training phase. Instead, it memorizes the training data and makes predictions by comparing new instances with stored ones.
KNN follows these steps to classify or predict a new data point:

1. **Choose the number of neighbors (K):**
   - The value of **K** determines how many nearest data points influence the prediction.
   - A small K (e.g., K=1) may lead to overfitting, while a large K can smooth the decision boundary.

2. **Compute the distance between the test sample and all training samples:**
   - Common distance metrics include:
     - **Euclidean Distance**: \( d(p, q) = \sqrt{\sum (p_i - q_i)^2} \)
     - **Manhattan Distance**: \( d(p, q) = \sum |p_i - q_i| \)
     - **Minkowski Distance**: Generalized form of both Euclidean and Manhattan distances.

3. **Sort training samples by distance:**
   - The closest K samples are selected.

4. **Determine the prediction based on K neighbors:**
   - **Classification**: Assign the most frequent class label among the K neighbors (majority vote).
   - **Regression**: Compute the average value of the K neighbors.

5. **Return the predicted label/value.**

## **4. Example: KNN for Classification**

### **Dataset**
| Height (cm) | Weight (kg) | Gender |
|------------|------------|--------|
| 170        | 70         | Male   |
| 160        | 60         | Female |
| 180        | 80         | Male   |
| 175        | 75         | Male   |
| 165        | 55         | Female |

**Test Data:** `Height = 172 cm, Weight = 68 kg`

### **Steps**
1. Compute distances from test point to all training points.
2. Select the `k=3` nearest points.
3. Determine the majority class among them.

### **Output**
Predicted **Gender: Male** (since most neighbors are male).

## **5. Example: KNN for Regression**

### **Dataset**
| Experience (Years) | Salary ($1000s) |
|--------------------|-----------------|
| 1                | 30              |
| 3                | 50              |
| 5                | 70              |
| 7                | 90              |
| 9                | 110             |

**Test Data:** `Experience = 4 years`

### **Steps**
1. Compute distances from the test point.
2. Select `k=3` nearest values.
3. Compute the average salary of selected values.

### **Output**
Predicted **Salary: $63.33k** (average of 50k, 70k, and 70k).

## Choosing the Right K Value
- **K=1:** May result in **overfitting** (high variance, sensitive to noise).
- **K too large:** Might lead to **underfitting** (high bias, less sensitivity to individual points).
- **Rule of thumb:** Choose \( K \approx \sqrt{n} \), where \( n \) is the dataset size.
- A common approach is using **cross-validation** to find the optimal `k`.
- Typically, an **odd K** (like 3, 5, 7) is chosen to avoid ties in classification.

## Advantages of KNN
✔ Simple to implement and understand.  
✔ Works well for small datasets and with multi-class classification. 
✔ No training phase; only requires storage of training data. 

## Disadvantages of KNN
❌ Slow for large datasets (computational cost grows with more samples).  
❌ Sensitive to irrelevant or redundant features.  
❌ Requires proper scaling (features should be normalized for Euclidean distance).
❌ Computationally expensive for large datasets.
❌ Performance depends on the choice of K and distance metric. 


