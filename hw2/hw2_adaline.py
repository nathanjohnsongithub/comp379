"""
    .py file for the Adaline implementation for question 3 and question 4 of hw2.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Implementation of adaptive linear neuron (Adaline) classifier from the textbook
class Adaline(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression, we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
# Lets preprocess the Titanic dataset and then split it into training and test sets
df = pd.read_csv('data/train.csv')

df = df.drop(columns=["Name", "Ticket", "Cabin"])  
df = df.dropna()

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Split into features and labels
X = df.drop(columns=["Survived"]).values
y = df["Survived"].values

# Adaline expects labels as -1 and 1 instead of 0 and 1
y = np.where(y == 0, -1, 1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=35, shuffle=True, stratify=y
)

# Initialize and train Adaline
adaline = Adaline(eta=0.0001, n_iter=100)
adaline.fit(X_train, y_train)

# Training progress
print("Final weights:", adaline.w_)

# Predictions
y_pred_train = adaline.predict(X_train)
y_pred_test = adaline.predict(X_test)

train_accuracy = (y_pred_train == y_train).mean()
test_accuracy = (y_pred_test == y_test).mean()

print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Print out the outputs with the features to see what is most predictive
feature_names = df.drop(columns=["Survived"]).columns
for name, weight in zip(feature_names, adaline.w_[1:]):  # skip bias term at index 0
    print(f"{name:11s} -> {weight:.3f}")

# Baseline model implementation with random weights
def random_weights_baseline(X):
    # Generate random weights
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

    # Generate predictions
    linear_output = np.dot(X, w) + b
    y_pred = np.where(linear_output >= 0, 1, -1)
    return y_pred

# Baseline accuracy for adaline
y_pred_baseline = random_weights_baseline(X_train)
baseline_accuracy = (y_pred_baseline == y_train).mean()

print()
print(f"Baseline (random weights) accuracy for adaline: {baseline_accuracy:.2f}")
print(f"Adaline accuracy on training set: {train_accuracy:.2f}")
