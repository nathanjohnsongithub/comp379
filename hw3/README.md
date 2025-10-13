To run the code you will need `pandas`, `numpy`, `scikit-learn`, and `ucimlrepo`. If you don't want to run the code from the .py files I've pasted the output below


| fixed_acidity | volatile_acidity | citric_acid | residual_sugar | chlorides |
|---------------|------------------|--------------|----------------|------------|
| 7.4           | 0.70             | 0.00         | 1.9            | 0.076      |
| 7.8           | 0.88             | 0.00         | 2.6            | 0.098      | 
| 7.8           | 0.76             | 0.04         | 2.3            | 0.092      |
| 11.2          | 0.28             | 0.56         | 1.9            | 0.075      | 
| 7.4           | 0.70             | 0.00         | 1.9            | 0.076      |

| free_sulfur_dioxide  | total_sulfur_dioxide | density |  pH   | sulphates  | alcohol |
|----------------------|----------------------|----------|------|------------|----------|
| 11.0                 | 34.0                 | 0.9978   | 3.51 | 0.56       | 9.4      |
| 25.0                 | 67.0                 | 0.9968   | 3.20 | 0.68       | 9.8      |
| 15.0                 | 54.0                 | 0.9970   | 3.26 | 0.65       | 9.8      |
| 17.0                 | 60.0                 | 0.9980   | 3.16 | 0.58       | 9.8      |
| 11.0                 | 34.0                 | 0.9978   | 3.51 | 0.56       | 9.4      |

```
Untuned Development set accuracy: 0.5436


Best SVM dev accuracy after tuning: 0.6113
Best hyperparameters combination: {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}

k= 1 | Dev Accuracy = 0.6144
k= 3 | Dev Accuracy = 0.5436
k= 5 | Dev Accuracy = 0.5549
k= 7 | Dev Accuracy = 0.5600
k= 9 | Dev Accuracy = 0.5682
k=11 | Dev Accuracy = 0.5641
k=13 | Dev Accuracy = 0.5723
k=15 | Dev Accuracy = 0.5600

Best k = 1 | Dev Accuracy = 0.6144

Strategy    : most_frequent
Dev Accuracy: 0.4205

Strategy    : stratified
Dev Accuracy: 0.3559

==============================
Dummy most_frequent classifier
Test Accuracy: 0.4164
Test F1: 0.0840

Best SVC Model {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}
Test Accuracy: 0.5723
Test F1: 0.2954

Best My KNN Model k=1
Test Accuracy: 0.6236
Test F1: 0.3885

=== Final Comparison ===
Model                  Dev Acc     Test Acc    Test F1
------------------------------------------------------------
Dummy (most_frequent)  0.4205      0.4164      0.0840          
Best SVC               0.6800      0.5723      0.2954          
Best My KNN            0.6144      0.6236      0.3885    
```

-----

**Note:** Some warnings pop up when I run telling me it may have not converged for the SVCs. So if you run it and you get the warnings, I have just excluded them from this output for simplicity