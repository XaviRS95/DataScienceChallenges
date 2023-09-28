import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv('balanced_training_data.csv')

y = data['loan_status']
X = data.drop('loan_status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(class_weight='balanced', n_estimators=150, bootstrap=False)
gb = GradientBoostingClassifier(random_state=1234)
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=2)

print('RandomForest:')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(roc_auc_score(y_pred, y_test))
print(precision_score(y_pred, y_test))
print('-------------------------------')

print('GradientBoost:')
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(roc_auc_score(y_pred, y_test))
print(precision_score(y_pred, y_test))
print('-------------------------------')