import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data/train.csv')

# Drop columns I think are unnecessary for predicting survival
df = df.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Embarked'])

# Convert Sex to 0 for male and 1 for female
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Weights for each feature to compute weighted average
weights = {
    'Pclass': 0.15,
    'Sex': 0.5,
    'Age': 0.3,
    'Fare': 0.05
}

def minmax_normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

weighted_averages = [0] * (len(df) + 1) # Initialize with an extra element because passenger IDs start at 1

# Precompute min and max for each column to avoid recalculating in the loop
col_min = {c: df[c].min() for c in weights.keys()}
col_max = {c: df[c].max() for c in weights.keys()}

# Compute the weighted average for each passenger
for index, row in df.iterrows():
    total = 0
    for col, weight in weights.items():
        norm_val = minmax_normalize(row[col], col_min[col], col_max[col])
        if col == 'Age' or col == 'Pclass':  # For Age and Pclass, lower values are better
            norm_val = 1 - norm_val  
        total += norm_val * weight

    weighted_averages[index + 1] = total

# Predict survival if weighted average >= 0.5
weighted_averages = weighted_averages[1:]  # drop the dummy 0th element
pred_labels = [1 if wa >= 0.5 else 0 for wa in weighted_averages]


# Now lets check how accurate the model is
y_true = list(df["Survived"])
n = len(y_true)

# Calculate accuracy
correct = sum(1 for yt, yp in zip(y_true, pred_labels) if yt == yp)
accuracy = correct / n

# Baseline accuracy (always predict the majority class)
num_dead = y_true.count(0)
num_alive = y_true.count(1)
baseline = max(num_dead, num_alive) / n

# How many predicted survive
num_predicted_survive = sum(pred_labels)

print(f"Number of passengers predicted to survive: {num_predicted_survive}")
print(f"Total number of passengers: {n}")
print(f"Proportion predicted to survive: {num_predicted_survive / n:.2f}")

# Actual survivors
actual_survivors = sum(y_true)
print(f"Actual number of survivors: {actual_survivors}")
print(f"Accuracy: {accuracy:.3f} (Baseline always majority class: {baseline:.3f})")
