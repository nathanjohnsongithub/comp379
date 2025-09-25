"""
.py file for the perceptron implementation for questions 1 and 2 of hw2.
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
 
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        # Convert to binary output
        return (np.asarray(x) >= 0).astype(int)
    
X = np.array([
    [ 2.0,  1.5],
    [ 2.2,  2.1],
    [ 1.8,  2.5],
    [ 2.5,  1.8],
    [ 3.0,  2.2],
    [-2.0, -1.5],
    [-2.2, -2.1],
    [-1.8, -2.5],
    [-2.5, -1.8],
    [-3.0, -2.2]
])

y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Initialize and train the perceptron
p = Perceptron(learning_rate=0.1, n_iterations=200)
p.fit(X, y)
print("Weights:", p.weights)
print("Bias:", p.bias)

plt.figure()
# Plot data points
plt.scatter(X[y==1,0], X[y==1,1], label="Class 1 (y=1)")
plt.scatter(X[y==0,0], X[y==0,1], label="Class 0 (y=0)")

# Calculate and plot decision boundary
w1, w2 = p.weights
b = p.bias

# Handle the case where w2 is zero to avoid division by zero
x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
if abs(w2) > 1e-12:
    x2_vals = -(w1/w2) * x1_vals - b / w2
    plt.plot(x1_vals, x2_vals, label="Decision boundary")
# If w2 is zero, the decision boundary is vertical
else:
    x_val = -b / w1 if abs(w1) > 1e-12 else 0.0
    plt.axvline(x=x_val, label="Decision boundary")

plt.title(f"Perceptron Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

# Code to evaluate the model
y_pred = p.predict(X)
accuracy = (y_pred == y).mean()
print(f"Training accuracy: {accuracy:.2f}")

X_NOT_SEPARABLE = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.5, 0.2],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.8, 0.9],
    [0.3, 0.4],
    [0.6, 0.7],
    [0.5, 0],
    [0, 0.5],
    [0.65, 0.2]
])

y_not_separable = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])

# Initialize and train the perceptron on non-linearly separable data
p_not_separable = Perceptron(learning_rate=0.1, n_iterations=500) # Sufficient iterations to observe non-convergence
p_not_separable.fit(X_NOT_SEPARABLE, y_not_separable)
print("Weights for non-linearly separable data:", p_not_separable.weights)
print("Bias for non-linearly separable data:", p_not_separable.bias)

plt.figure()
# Plot data points
plt.scatter(X_NOT_SEPARABLE[y_not_separable==1,0], X_NOT_SEPARABLE[y_not_separable==1,1], label="Class 1 (y=1)")
plt.scatter(X_NOT_SEPARABLE[y_not_separable==0,0], X_NOT_SEPARABLE[y_not_separable==0,1], label="Class 0 (y=0)")

# Calculate and plot decision boundary
w1, w2 = p_not_separable.weights
b = p_not_separable.bias

# Handle the case where w2 is zero to avoid division by zero
x1_vals = np.linspace(X_NOT_SEPARABLE[:,0].min()-1, X_NOT_SEPARABLE[:,0].max()+1, 200)
if abs(w2) > 1e-12:
    x2_vals = -(w1/w2) * x1_vals - b / w2
    plt.plot(x1_vals, x2_vals, label="Decision boundary")
# If w2 is zero, the decision boundary is vertical
else:
    x_val = -b / w1 if abs(w1) > 1e-12 else 0.0
    plt.axvline(x=x_val, label="Decision boundary")

plt.title(f"Perceptron Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

# Baseline model implementation with random weights
def random_weights_baseline(X):
    # Generate random weights
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

    # Generate predictions
    linear_output = np.dot(X, w) + b
    y_pred = np.where(linear_output >= 0, 1, -1)
    return y_pred

# Code to evaluate the model
y_pred_not_separable = p_not_separable.predict(X_NOT_SEPARABLE)
accuracy_not_separable = (y_pred_not_separable == y_not_separable).mean()
print(f"Training accuracy for non-linearly separable data: {accuracy_not_separable:.2f}")

# Baseline accuracy for perceptron 
y_pred_baseline_perceptron = random_weights_baseline(X)
baseline_accuracy_perceptron = (y_pred_baseline_perceptron == y).mean()

# Display findings
print()
print(f"Baseline (random weights) accuracy for perceptron: {baseline_accuracy_perceptron:.2f}")
print(f"Perceptron accuracy for linearly separable data: {accuracy:.2f}")
print(f"Perceptron accuracy for non-linearly separable data: {accuracy_not_separable:.2f}")
