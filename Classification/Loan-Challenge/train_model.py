import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, classification_report


data = pd.read_csv('balanced_training_data.csv')

y = data['loan_status']
X = data.drop('loan_status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Find the best estimator for our RandomForest model:
custom_scorer = make_scorer(roc_auc_score, greater_is_better=True)
param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],  # being the number of trees in the forest.
                  'min_samples_leaf': [3, 5, 10, 20],  # number of minimum samples required at a leaf node.
                  'min_samples_split': [3, 6, 9],  # number of minimum samples required to split an internal node.
                  'criterion': ['entropy'],  # measures the quality of a split. Can use gini's impurity or entropy.
                  # 'subsample':[0.5,0.8,1]#buscar con mas detalle
                  # 'reg_lambda':[1,10,100]#buscar con mas detalle
                  }

clf = GridSearchCV(
# Evaluates the performance of different groups of parameters for a model based on cross-validation.
    RandomForestClassifier(class_weight='balanced', bootstrap=False, random_state=1234),
    param_grid,  # dict of parameters.
    cv=10,  # Specified number of folds in the Cross-Validation(K-Fold).
    scoring='f1_micro')

clf.fit(X_train, y_train)

print(clf.best_estimator_)
model = clf.best_estimator_  # Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data


#Roc_Auc_Score was used because in the original challenge, it was the established metric.
print('RandomForest:')
y_pred = model.predict(X_test)
print(roc_auc_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
print('-------------------------------')