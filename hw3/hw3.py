""" First lets load the wine quality dataset. """
# import fetch_ucirepo function for fetching datasets (cool)
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
    
print(X.head(), "\n")


""" Now lets preprocess the data into 70% training, 15% development, and 15% test sets and standardize the features. """ 
# Split the data into training, development, and testing sets
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, random_state=19)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5, random_state=19)

# Convert the target data to 1d numpy arrays
y_train = np.ravel(y_train)
y_dev = np.ravel(y_dev)
y_test = np.ravel(y_test)

# Standardize the features using Sklearn
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


""" Finally, lets train a logistic regression model and evaluate it on the development set. """
# Train a SVC classifier with default hyperparameters
clf = SVC(kernel="linear", C=1.0, random_state=19)
clf.fit(X_train, y_train)

# Evaluate
y_dev_pred = clf.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Untuned Development set accuracy: {dev_accuracy:.4f}")


""" Now lets see if we can improve the model by doing hyperparameter tuning using the development set. """
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
            model.fit(X_train, y_train)
            acc = accuracy_score(y_dev, model.predict(X_dev))
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {"kernel": kernel, "C": C}
        else:  # "rbf"
            for gamma in svm_grid["gamma"]:
                model = SVC(kernel="rbf", C=C, gamma=gamma, max_iter=5000, tol=1e-3, verbose=False)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_dev, model.predict(X_dev))
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {"kernel": kernel, "C": C, "gamma": gamma}

print(f"\nBest SVM dev accuracy after tuning: {best_acc:.4f}")
print(f"Best hyperparameters combination: {best_params}\n")


""" Now let me make my own KNN class from scratch and compare my performance with sklearn's KNN. """
class The_Better_KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _compute_distances(self, X):
        """
        Compute Euclidean distances between each test sample and all training samples.
        """

        # Get the number of test and training samples
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Create an empty distance matrix (test_samples x train_samples)
        dists = np.zeros((num_test, num_train))

        # Loop through all the samples
        for i in range(num_test):
            for j in range(num_train):
                # Compute the Euclidean distance manually:
                diff = X[i] - self.X_train[j]
                squared_diff = diff ** 2
                sum_squared_diff = np.sum(squared_diff)
                distance = np.sqrt(sum_squared_diff)

                # Save the distance in the matrix
                dists[i, j] = distance

        # Return the full distance matrix
        return dists

        
    def predict(self, X):
        """
        Predict labels for each test sample using the KNN algorithm.
        """
        # First Compute the distances from each test sample to every training sample
        dists = self._compute_distances(X)

        # Find the indices of the k nearest neighbors for each test sample by sorting the distances
        knn_indices = np.argsort(dists, axis=1)[:, :self.k]

        # Get the corresponding labels of those nearest neighbors
        knn_labels = self.y_train[knn_indices]

        # Finally, for each test sample, choose the most common (majority) label
        preds = np.array([np.bincount(row).argmax() for row in knn_labels])

        # Return the predicted labels
        return preds


""" Now that I've implemented my own KNN class, lets tune it to see what the best value for k is. """
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_acc_my = -1

# Try all k values
for k in k_values:
    model = The_Better_KNN(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)
    acc = accuracy_score(y_dev, preds)
    print(f"k={k:2d} | Dev Accuracy = {acc:.4f}")

    if acc > best_acc_my:
        best_acc_my = acc
        best_k = k

print(f"\nBest k = {best_k} | Dev Accuracy = {best_acc_my:.4f}\n")


""" Lets see how a baseline model does so we can compare it to all the different models we've tested and tuned. """
strategies = ["most_frequent", "stratified"]

for strat in strategies:
    dummy = DummyClassifier(strategy=strat, random_state=19)
    dummy.fit(X_train, y_train)

    # Evaluate on development set
    y_dev_pred = dummy.predict(X_dev)
    acc = accuracy_score(y_dev, y_dev_pred)

    print(f"Strategy    : {strat}")
    print(f"Dev Accuracy: {acc:.4f}\n")


""" Finally, lets evaluate the best model on the test set. """
# Evaluate best DummyClassifier on test set
best_dummy = DummyClassifier(strategy="most_frequent", random_state=19)
best_dummy.fit(X_train, y_train)

y_test_pred_dummy = best_dummy.predict(X_test)
dummy_test_acc = accuracy_score(y_test, y_test_pred_dummy)
dummy_test_f1 = f1_score(y_test, y_test_pred_dummy, average="macro")

print("Dummy most_frequent classifier")
print(f"Test Accuracy: {dummy_test_acc:.4f}")
print(f"Test F1: {dummy_test_f1:.4f}\n")

# Evaluate best Sklearn KNN model on test set
best_SVC = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    max_iter=5000,
)
best_SVC.fit(X_train, y_train)

y_test_pred_SVC = best_SVC.predict(X_test)
SVC_test_acc = accuracy_score(y_test, y_test_pred_SVC)
SVC_test_f1 = f1_score(y_test, y_test_pred_SVC, average="macro")

print(f"Best SVC Model {best_params}")
print(f"Test Accuracy: {SVC_test_acc:.4f}")
print(f"Test F1: {SVC_test_f1:.4f}\n")

# Evaluate best My KNN model on test set
best_my_knn = The_Better_KNN(k=best_k)
best_my_knn.fit(X_train, y_train)
y_test_pred_my_knn = best_my_knn.predict(X_test)
my_knn_test_acc = accuracy_score(y_test, y_test_pred_my_knn)
my_knn_test_f1 = f1_score(y_test, y_test_pred_my_knn, average="macro")

print(f"Best My KNN Model k={best_k}")
print(f"Test Accuracy: {my_knn_test_acc:.4f}")
print(f"Test F1: {my_knn_test_f1:.4f}\n")

print("=== Final Comparison ===")
print(f"{'Model':<23}{'Dev Acc':<12}{'Test Acc':<12}{'Test F1':<12}")
print(f"{'-'*60}")
print(f"{'Dummy (most_frequent)':<23}{0.4205:<12.4f}{dummy_test_acc:<12.4f}{dummy_test_f1:<12.4f}")
print(f"{'Best SVC':<23}{0.6800:<12.4f}{SVC_test_acc:<12.4f}{SVC_test_f1:<12.4f}")
print(f"{'Best My KNN':<23}{best_acc_my:<12.4f}{my_knn_test_acc:<12.4f}{my_knn_test_f1:<12.4f}")
