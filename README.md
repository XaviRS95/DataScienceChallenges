# Machine Learning Challenges

This is a collection of all the challenges that I will take in order to learn and grasp all the concepts in the areas of Machine and Deep Learning. 

## Classification Challenges:

### Horses Health:

Categorical competition from Kaggle. You have to predict the outcome of a horse (lived, died or euthanized) using the provided dataset using f1-score. 
[Original Horses Health Kaggle Challenge][1]<br/>
Additional data from the original dataset of the challenge was also included to improve the model's performance [Original horses data][2]

Notebook can be accessed [here][3]

This dataset contained 1534 rows in total with 29 columns.

Used libraries:
- Pandas
- ImbLearn
- Scikit-Learn

Techniques applied to the data:
- Fill NaN values.
- Merge classes with very few rows to reduce number of classes in a feature.
- Encode categorical variables.
- MinMax Scale numerical variables.
- OverSample minority class 0 for class 1 (Majority class was way bigger to generate random)

Multiple models were evaluated, but the model with highest f1-score was RandomForest. 

Techniques applied for Model-Tuning:
- Data was split in 80%/20% for training/validation.
- GridSearchCV was applied for 10 folds.

#### RandomForestClassifier:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.89      | 0.78   | 0.83     |
| 1     | 0.94      | 0.88   | 0.91     |
| 2     | 0.70      | 0.87   | 0.77     |

Overall Performance

| Metric           | Value |
|------------------|-------|
| **Accuracy**      | 0.84  |
| **Macro Avg Precision**  | 0.84  |
| **Macro Avg Recall**     | 0.84  |
| **Macro Avg F1-Score**   | 0.84  |
| **Weighted Avg Precision** | 0.85 |
| **Weighted Avg Recall**    | 0.84 |
| **Weighted Avg F1-Score**  | 0.84 |
| **Total Support**        | 452   |


### Loan Challenge:
https://github.com/XaviRS95/DataScienceChallenges/tree/main/Classification/Loan-Challenge
Binary classification competition from Hacker Earth where you need to find out if a bank client that asked for a loan will default it or not. For this, a dataset of 600k+ rows is provided and roc-accuracy was the metric used to determine the winner.

### 2912 Titanic:
https://github.com/XaviRS95/DataScienceChallenges/tree/main/Classification/Titanic-2912


### Titanic:
https://github.com/XaviRS95/DataScienceChallenges/tree/main/Classification/Titanic

[1]: https://github.com/XaviRS95/DataScienceChallenges/tree/main/Classification/Horses-Health "Original Horses Health Kaggle Challenge"
[2]: https://www.kaggle.com/datasets/yasserh/horse-survival-dataset "Original horses data"
[3]: https://github.com/XaviRS95/DataScienceChallenges/blob/main/Classification/Horses-Health/Approach_RandomForestClassifier.ipynb "Horses RandomForestClassifier Approach"