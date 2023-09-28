import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

def get_top_important_features_list(coefficient, columns, importances,clf):
  '''Obtiene las características más importantes en orden descendente'''
  indices = np.argsort(importances)[::-1]  # Indices de las características mas significativas
  feature_names = [columns[x] for x in indices]  # Nombre de las características
  mif = importances[indices[0]]  # Valor de la característica más importante
  features = [columns[x] for x in indices if importances[x] >= mif * (1 - coefficient)]
  print("Number of selected features:", len(features), "from a total of:", len(columns))
  print("Most important feature is:", features[0], "in position", indices[0], "with importance of ", mif)
  print("Features included:",features)
  show_ranked_features(clf=clf, columns_num=len(columns), importances=importances, indices=indices, feature_names=feature_names)
  return features

def show_ranked_features(clf, columns_num, importances, indices, feature_names):
  # Gráfica mostrando el ranking de características de más a menos importante
  std = np.std([tree.feature_importances_ for tree in clf.best_estimator_.estimators_], axis=0)
  plt.figure(figsize=(20, 20))
  plt.title("Feature importances")
  plt.barh(range(columns_num), importances[indices], color="b", xerr=std[indices], align="center")
  plt.yticks(range(columns_num), feature_names)
  plt.ylim([columns_num, -1])
  plt.savefig('features_loan.png')


param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],  # being the number of trees in the forest.
                  'min_samples_leaf': [3],  # number of minimum samples required at a leaf node.
                  'min_samples_split': [6],  # number of minimum samples required to split an internal node.
                  'criterion': ['entropy']
                  }


custom_scorer = make_scorer(accuracy_score, greater_is_better=True)
clf = GridSearchCV(
            # Evaluates the performance of different groups of parameters for a model based on cross-validation.
            RandomForestClassifier(class_weight='balanced', bootstrap=False),
            param_grid,  # dict of parameters.
            cv=10,  # Specified number of folds in the Cross-Validation(K-Fold).
            scoring=custom_scorer)


data = pd.read_csv('feature_selection_data.csv')
y = data['loan_status']
X = data.drop('loan_status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf.fit(X_train, y_train)
model = clf.best_estimator_

importances = model.feature_importances_
#List of top % important features in the model are obtained. This % regulated by coefficient between [0,1].

features = get_top_important_features_list(clf=clf, coefficient=0.90, columns=X_train.columns, importances=importances)
