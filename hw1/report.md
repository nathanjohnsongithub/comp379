# Comp 379 - HW 1 Report

## Outline and Justification

My goal was to implement what was suggested from the homework document. This was essentially assigning different weights to each of the features and then computing the weighted average of each of the columns. After computing the weighted average you would get a value in the range of 0 to 1. If this value was `> 0.5` we would say that the passenger survived the sinking (represented as a `1`), if the value was `< 0.5` we would say that the passenger died during the sinking (represented as a `0`).

For the model, I decided to keep only some of the features from the original dataset, I kept `Pclass`, `Sex`, `Age`, and `Fare`. All of the other attributes I thought wouldn't be as helpful to predict survival, or at least I wouldn't know what weight to set for it. For example, I can't see how cabin number would be overly important for survival, compared to something like age, where younger people are preferred over older.

The real "magic here" is the weights we assigned to each of the features or attributes. (And by magic I mean magic numbers I made up). Below is the respective weights for each of the attributes.

```python
weights = {
    'Pclass': 0.15,
    'Sex': 0.5,
    'Age': 0.3,
    'Fare': 0.05
} 
```
- **Pclass**:  This is the ticket class the passenger was in with `1 = 1st, 2 = 2nd, 3 = 3rd`. It's essentially an indicator for the socio-economic status for the passenger. I'm assuming here that if you were more wealthy during the sinking, you had slight priority to be saved.
- **Sex**: The sex of the given passenger. I've heard before that women were preferred over men heavily for who was saved first, so this is 50% of our prediction. 
- **Age**: How old the given passenger was. I've also heard that if you were a kid you were preferred over any older people so this is 30% of our prediction. 
- **Fare**: This is how much the ticket cost for the given passenger. This is very similar to me as the `Pclass` field so this has a very small weight, because I believe it's doing similar stuff as Pclass.

Basically, If you're young and a women and rich we predict that you would survive. If you're old, a man, and poor we predict that you died. All of these percentages are just randomly assigned based on what I think would be the most accurate from things I've heard about the titanic (like women and children being saved).

Didn't decide to use the test data because it doesn't have anything to measure the accuracy off of. I could implement it to predict the survival for each passenger, but I have no way to know how accurate that is so I decided to omit it for now.

## Steps to Implement
1. **Load data**
   - Read `train.csv` into a pandas dataframe.

2. **Select & clean columns**
   - Drop: `Name, Ticket, SibSp, Parch, Cabin, Embarked` 
   - Encode `Sex` → `{male: 0, female: 1}` 

3. **Normalize using training stats**
   - Compute `min`/`max` for `Pclass, Sex, Age, Fare` which we need for normalization.
   - Apply min–max normalization to training data.

4. **Scoring rule**
   - Use the weights I assigned and explained above weights: `Pclass: 0.15, Sex: 0.5, Age: 0.3, Fare: 0.05`.
   - After normalization we invert the value for `Age` and `Pclass` because lower is “better”,  `1 - norm_val`.
   - Sum weighted values to get a score per passenger.

5. **Predict**
   - Label `1` (survived) if score >= 0.5 otherwise `0`.

6. **Evaluate on train**
   - Compare predictions to `train["Survived"]` to compute accuracy and majority-class baseline.

7. **Report findings**
   - Print out the results, with number of passengers predicted to survive, the actual number of survivors, and the accuracy compared to the baseline. 


**Note**: *I have a good amount of experience with Pandas before this and even more experience with python so that helped a lot for this assignment.*

## Results

- **Number predicted to survive:** 261  
- **Total passengers:** 891  
- **Proportion predicted to survive:** 0.29  
- **Actual survivors:** 342  
- **Accuracy:** **0.765**  
- **Baseline (always predict majority class):** **0.616**

**Interpretation:**
- This predicts better than only guessing dead by about 14.9%, which I will take.
- I don't fully know what a good value would be for this, but the fact I made a "model" that predicts something better then random makes me proud.
- It seems that this model under predicts the amount of survivors. (261 predicted vs 342 actual). This may mean that the 0.5 threshold may be too harsh. Lowering the threshold slightly may improve the accuracy of the model.
