# Nathan Johnson Machine Learning HW3

## 1st, load the wine set data and split it into train, test and development sets then standardize it. Below is what the train data looks like. The goal is to guess the type of wine based on the chemical make up of it

| fixed_acidity | volatile_acidity | citric_acid | residual_sugar | chlorides |
|---------------|------------------|--------------|----------------|------------|
| 7.4           | 0.70             | 0.00         | 1.9            | 0.076      |
| 7.8           | 0.88             | 0.00         | 2.6            | 0.098      | 
| 7.8           | 0.76             | 0.04         | 2.3            | 0.092      |
| 11.2          | 0.28             | 0.56         | 1.9            | 0.075      | 
| 7.4           | 0.70             | 0.00         | 1.9            | 0.076      |

| free_sulfur_dioxide | total_sulfur_dioxide | density | pH   | sulphates | alcohol |
|----------------------|----------------------|----------|------|------------|----------|
| 11.0                 | 34.0                 | 0.9978   | 3.51 | 0.56       | 9.4      |
| 25.0                 | 67.0                 | 0.9968   | 3.20 | 0.68       | 9.8      |
| 15.0                 | 54.0                 | 0.9970   | 3.26 | 0.65       | 9.8      |
| 17.0                 | 60.0                 | 0.9980   | 3.16 | 0.58       | 9.8      |
| 11.0                 | 34.0                 | 0.9978   | 3.51 | 0.56       | 9.4      |


## Now lets train a baseline SVC model and see how it performs
```
Untuned Development set accuracy: 0.5436
```

## Lets now tune this SVC model and check the accuracy now. Below is the SVM grid we used for tuning
``` python
svm_grid = {
    "kernel": ["linear", "rbf"], # We didn't talk about rbf a ton but I wanted to try it because its very cool
    "C": [0.1, 1, 10, 100],          
    "gamma": ["scale", "auto"], # only used for rbf
}
```
and here is best SVM after tuning
```
Best SVM dev accuracy after tuning: 0.6113
Best hyperparameters combination: {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
```

## Next, the HW asks to make your own KNN and find the best value for k. The output for that code is below. You can check hw3.py for the implementation
```
k= 1 | Dev Accuracy = 0.6144
k= 3 | Dev Accuracy = 0.5436
k= 5 | Dev Accuracy = 0.5549
k= 7 | Dev Accuracy = 0.5600
k= 9 | Dev Accuracy = 0.5682
k=11 | Dev Accuracy = 0.5641
k=13 | Dev Accuracy = 0.5723
k=15 | Dev Accuracy = 0.5600
```

As you can see the best value for k was one with an accuracy for 0.6144. 
`Best k = 1 | Dev Accuracy = 0.6144`

## Now the HW wanted us to train some baseline models for comparison. 

I trained a most_frequent baseline model and a stratified baseline model. Below is the accuracy from each of those.
```
Strategy    : most_frequent
Dev Accuracy: 0.4205
```
```
Strategy    : stratified
Dev Accuracy: 0.3559
```

## Finally, I was asked to take the best version of the SVM and KNN and compare them to the baselines on the testing dataset. Below is the output and a table for comparison
```
Dummy most_frequent classifier
Test Accuracy: 0.4164
Test F1: 0.0840
```
```
Best SVC Model {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
Test Accuracy: 0.5723
Test F1: 0.2954
```
```
Best My KNN Model k=1
Test Accuracy: 0.6236
Test F1: 0.3885
```
| Model                 | Dev Acc | Test Acc | Test F1  |
|------------------------|---------|-----------|--------------|
| Dummy (most_frequent)  | 0.4205  | 0.4164    | 0.0840       |
| Best SVC               | 0.6800  | 0.5723    | 0.2954       |
| Best My KNN            | 0.6144  | 0.6236    | 0.3885       |

## Summarizing my findings
Overall, I found that both SVM and KNN outperformed the random baseline models by a significant amount. This suggests that some meaningful patterns exist within the wine dataset. After performing grid search tuning for the SVM the highest performing was an RBF kernel with `C=10` and `gamma='scale'`. This achieved a development accuracy of `0.6113` and a test accuracy of `0.5723`. My custom KNN implementation performed slightly better, with the best result occurring interestingly enough at k=1. It had a development accuracy of `0.6144` and a test accuracy of `0.6236`. In comparison, the most_frequent baseline achieved only 0.4164 accuracy. While SVM captured more complex non-linear boundaries because of the RBF kernel KNN’s simplicity seemed to generalize slightly better for the wine dataset.

I also computed the F1 score to help evaluate how fairly each classifier performed across the different wine categories. The best KNN model `(k=1)` achieved a macro F1 of `0.3885`, outperforming the SVM model `0.2954` and the most_frequent baseline `0.0840`. This shows that KNN not only achieved higher overall accuracy but also maintained better balance. Overall, KNN proved to be the most effective model for this dataset, outperforming both the tuned SVM and random baselines.
