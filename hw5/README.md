To run the code you will need `pandas`, `numpy`, `scikit-learn`, and `ucimlrepo`. If you don't want to run the code from the .py files I've pasted the output below

---

```
SVC 10-Fold Mean Accuracy: 0.9750
SVC 10-FoldScores: ['1.0000', '1.0000', '1.0000', '1.0000', '1.0000', 
'1.0000', '1.0000', '0.8333', '0.9167', '1.0000']


Best SVM dev accuracy after tuning: 0.9750
Best hyperparameters combination: {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}

Best SVC Model {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
Test Accuracy: 0.9333
Test F1: 0.9267

--- Comparison between Train and Test Performance for Best SVC Model ---
Train Accuracy   : 1.0000 | Test Accuracy:  0.9333
Train F1         : 1.0000 | Test F1:        0.9267
```