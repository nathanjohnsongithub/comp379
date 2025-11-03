# Nathan Johnson Comp 379 Homework 5

### Dataset and Preprocessing

For this assignment, I used the Iris dataset which was briefly mentioned in class. The dataset consists of 150 samples of iris flowers with 4 features (sepal length, sepal width, petal length, petal width) and labeled as one of three species. Since the features are measured on different scales I applied standardization using StandardScaler.

### Model and Cross-Validation

I chose Support Vector Machines (SVM) for classification because they perform pretty well on small and separated datasets such as the Iris dataset. Per the requirements I made my own n-cross fold validation function for tuning the model. This function randomly shuffles the data, splits it into n folds, and iteratively trains on n-1 folds while testing on the remaining fold. Each fold is used once as a validation set, and the average accuracy across all folds is returned. Using 10-fold cross-validation, the base SVM with a linear kernel achieved an average accuracy of `0.975`.

### Grid Search for Hyperparameter Tuning

To find the best hyperparameters, I implemented a manual grid search over:

- kernel = {`linear`, `rbf`}
- C = {`0.1`, `1`, `10`, `100`} 
- gamma = {`scale`, `auto`} *(for RBF kernel only)*

The kernel determines how the model separates data. The `linear` kernels create straight decision boundaries, while nonlinear ones like `RBF` can capture more complex patterns. The `C` parameter controls how strictly the model penalizes misclassifications. A small `C` allows more errors but generalizes better whereas a large `C` fits the training data more tightly. The `gamma` value, which is only used in `RBF` kernels, controls how far the influence of a single data point extends. Gamma=`'scale'` automatically adjusts the kernel’s sensitivity based on the data’s variance, making it more balanced and less prone to overfitting than gamma=`'auto'` which uses a fixed value and can overreact to noise.

Each of the combinations was evaluated using my cross-validation function. The best configuration was found to be: 
```
Kernel: RBF, C: 10, Gamma: scale
``` 
This combination had a mean cross-validation accuracy of `0.975`.

### Final Evaluation and thoughts

The best SVM model was trained and evaluated on the test set. It scored a `0.9333` on the accuracy benchmark and `0.9267` for the F1 macro benchmark. This compared to the accuracy value on `1.0` on the training set and a F1 macro value of `1.0` shows a small drop in performance between training and test accuracy which suggest some slight overfitting. This is overall expected when tuning hyperparameters on a smaller dataset such as the Iris set. This is still an overall high performance metric which demonstrates a good generalization performance

I wanted to use Accuracy and F1 for evaluation metrics because the Iris dataset is quite balanced across its three classes, making accuracy a clear measure of overall correctness. The F1 score captures how well the model balances precision and recall for each class, ensuring strong performance even if slight misclassifications occur.