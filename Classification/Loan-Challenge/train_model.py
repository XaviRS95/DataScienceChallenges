import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('balanced_training_data.csv')

y = data['loan_status']
X = data.drop('loan_status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(class_weight='balanced', n_estimators=150, bootstrap=False)

print('RandomForest:')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(roc_auc_score(y_pred, y_test))
print('-------------------------------')