""" I'll use the Iris dataset for HW5 """
# import fetch_ucirepo function for fetching datasets (cool)
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

# Set random seed for reproducibility
np.random.seed(21)

# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
    
""" Now lets preprocess the data into 80% training, 10% development, and 10% test sets """ 
# Split the data into training, development, and testing sets
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=21)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5, random_state=21)

# Convert the target data to 1d numpy arrays
y_train = np.ravel(y_train)
y_dev = np.ravel(y_dev)
y_test = np.ravel(y_test)

""" Now lets standardize the features so they have mean 0 and variance 1. """
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


""" Now I will implement my own n-fold cross validation """
def n_fold_cross_validation(model, X, y, n_folds=5) -> tuple:
    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Split into n_folds folds
    fold_sizes = np.full(n_folds, len(X) // n_folds, dtype=int)
    fold_sizes[:len(X) % n_folds] += 1  # distribute the remaining sample if its not perfectly divisible by n_folds
    current = 0
    scores = []
    
    # Loop over each fold
    for fold_size in fold_sizes:
        # Create train and test sets
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        
        # Get the data
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train and score
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        # Move to next fold
        current = stop

    # Return mean score and all scores
    return np.mean(scores), scores


""" Lets test out my cross validation function with a simple SVM classifier. """
# Train a SVC classifier with default hyperparameters
model = SVC(kernel="linear", C=1.0, random_state=21)
mean_score, scores = n_fold_cross_validation(model, X_train, y_train, 10)
print(f"SVC 10-Fold Mean Accuracy: {mean_score:.4f}")
print(f"SVC 10-FoldScores: {[f'{score:.4f}' for score in scores]}\n")


""" Now let me implement by own grid search for hyperparameter tuning of SVM.
    I will test the different combinations using my n-fold cross validation function. """
# svm hyperparameter grid
svm_grid = {
    "kernel": ["linear", "rbf"], # We didn't talk about rbf a ton but I wanted to try it because its very cool
    "C": [0.1, 1, 10, 100],          
    "gamma": ["scale", "auto"], # only used for rbf
}

best_model = None
best_acc = -1.0
best_params = None

for kernel in svm_grid["kernel"]:
    for C in svm_grid["C"]:
        if kernel == "linear":
            # Using LinearSVC because when I would run SVC(kernel="linear") it would take forever
            model = LinearSVC(C=C, max_iter=5000, tol=1e-3)
            mean_score, scores = n_fold_cross_validation(model, X_train, y_train, n_folds=10)
            if mean_score > best_acc:
                best_acc = mean_score
                best_model = model
                best_params = {"kernel": kernel, "C": C}
        else:  # "rbf"
            for gamma in svm_grid["gamma"]:
                model = SVC(kernel="rbf", C=C, gamma=gamma, max_iter=5000, tol=1e-3, verbose=False)
                mean_score, scores = n_fold_cross_validation(model, X_train, y_train, n_folds=10)
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_model = model
                    best_params = {"kernel": kernel, "C": C, "gamma": gamma}

print(f"\nBest SVM dev accuracy after tuning: {best_acc:.4f}")
print(f"Best hyperparameters combination: {best_params}\n")


""" Now that we have the best hyperparameters for SVM, we can evaluate it on the development set. """
best_SVC = SVC(
    kernel=best_params["kernel"],
    C=best_params["C"],
    gamma=best_params.get("gamma", "scale"),
    max_iter=5000,
)

# Test the best model with the test set
best_SVC.fit(X_train, y_train)
y_test_pred_SVC = best_SVC.predict(X_test)
SVC_test_acc = accuracy_score(y_test, y_test_pred_SVC)
SVC_test_f1 = f1_score(y_test, y_test_pred_SVC, average="macro")

print(f"Best SVC Model {best_params}")
print(f"Test Accuracy: {SVC_test_acc:.4f}")
print(f"Test F1: {SVC_test_f1:.4f}\n")

# Now lets compare the SVC test set to the training set performance
y_train_pred_SVC = best_SVC.predict(X_train)
SVC_train_acc = accuracy_score(y_train, y_train_pred_SVC)
SVC_train_f1 = f1_score(y_train, y_train_pred_SVC, average="macro")


# Final comparison
print("--- Comparison between Train and Test Performance for Best SVC Model ---")
print(f"{'Train Accuracy':<17}: {SVC_train_acc:.4f} {'| Test Accuracy:':<17} {SVC_test_acc:.4f}")
print(f"{'Train F1':<17}: {SVC_train_f1:.4f} {'| Test F1:':<17} {SVC_test_f1:.4f}\n")
