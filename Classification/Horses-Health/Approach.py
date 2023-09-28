import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, classification_report
data = pd.read_csv('train.csv')

target = data['outcome']
data.drop(['id', 'hospital_number', 'abdomen'], axis = 1, inplace = True)

data['temp_of_extremities'].fillna('cool', inplace = True)
data['mucous_membrane'].fillna('other', inplace = True)
data['mucous_membrane'] = data['mucous_membrane'].apply(lambda x: 'other' if x in ['absent', 'increased'] else x)
data['capillary_refill_time'].fillna('less_3_sec', inplace = True)
data['capillary_refill_time'] = data['capillary_refill_time'].str.replace(pat = '3', repl = 'less_3_sec')
data['pain'].fillna('pain', inplace = True)
data['pain'] = data['pain'].str.replace(pat = 'slight', repl = 'severe_pain')
data['peristalsis'].fillna('absent', inplace = True)
data['abdominal_distention'].fillna('severe', inplace = True)
data['nasogastric_tube'].fillna('missing', inplace = True)
data['nasogastric_reflux'] = data['nasogastric_reflux'].str.replace(pat = 'slight', repl = 'missing')
data['nasogastric_reflux'].fillna('missing', inplace = True)
data['rectal_exam_feces'] = data['rectal_exam_feces'].str.replace(pat = 'serosanguious', repl = 'missing')
data['rectal_exam_feces'].fillna('missing', inplace = True)
data['abdomo_appearance'].fillna('missing', inplace = True)

float_columns = data.select_dtypes(include=['float64']).columns
object_columns = data.select_dtypes(include=['object']).columns

for column in float_columns:
    data[column].fillna(data[column].median(skipna = True), inplace = True)

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(data[float_columns])
data[float_columns] = minmax_scaler.transform(data[float_columns])

label_encoders = {x: LabelEncoder() for x in object_columns}

for column in object_columns:
    label_encoders[column].fit(data[column])
    data[column] = label_encoders[column].transform(data[column])


#####DATA WAS CLEARED AND READY TO BE USED. FROM HERE, THE TEST IS TO SEE IF THESE VARIABLES GET > 74% F1-SCORE ACCURACY

to_drop_cols = ['nasogastric_reflux_ph',
'pain',
'nasogastric_reflux',
'mucous_membrane',
'lesion_1',
'rectal_temp',
'lesion_3',
'lesion_2',
'cp_data',
'nasogastric_tube'
]


data.drop(columns=to_drop_cols, inplace=True)



data = data[['abdomo_appearance',
'pulse',
'capillary_refill_time',
'surgical_lesion',
'packed_cell_volume',
'abdominal_distention',
'total_protein',
'peristalsis',
'rectal_exam_feces',
'temp_of_extremities',
'outcome'
]]

new_data = pd.get_dummies(data, columns = ['abdomo_appearance','capillary_refill_time','surgical_lesion', 'abdominal_distention', 'peristalsis', 'rectal_exam_feces', 'temp_of_extremities'], dtype=np.int8)


y_to_oversample = new_data.query('outcome != 2')['outcome']
features_to_oversample = new_data.query('outcome  != 2').drop('outcome', axis = 1)

ros = RandomOverSampler(random_state=42)

X_res, y_res = ros.fit_resample(features_to_oversample, y_to_oversample)
label_encoders['outcome'].fit(y_res)
y_res = label_encoders['outcome'].transform(y_res)


new_data.query('outcome == 2', inplace = True)
X_res['outcome'] = y_res
new_data = pd.concat([new_data, X_res])


y = new_data['outcome']
new_data.drop('outcome', axis = 1, inplace = True)


X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=0.2)
#Because of the low number of rows, bootstrap will be set to False.
rf = RandomForestClassifier(class_weight='balanced', n_estimators=225, bootstrap=False)
gb = GradientBoostingClassifier(random_state=1234)
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=2)

print('RandomForest:')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_pred, y_test))
print('-------------------------------')


print('GradientBoost:')
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(classification_report(y_pred, y_test))
print('-------------------------------')

