import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data = pd.read_csv('train.csv')

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


corr = data.corr()
corr['columns'] = data.columns
corr.to_csv('correlation_non_dummy.csv', index = False)

y = data['outcome']
data.drop(columns = 'outcome', inplace=True)

data = pd.get_dummies(data, columns = object_columns.drop('outcome'), dtype=np.int8)

data['outcome'] = y
corr = data.corr()
corr['columns'] = data.columns
corr.to_csv('correlation_dummy.csv', index = False)