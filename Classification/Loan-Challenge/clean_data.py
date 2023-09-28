import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



###Function used to eliminate the categorical features that have been eliminated for the specified reasons.
# Returns only the categorical features that will be used.###
def clean_categorical_data(data):
  # Binary variable with 8 value of 'y' and 500k+ of 'n'
  data.drop(['pymnt_plan', 'application_type'], axis=1, inplace=True)
  # Droping due to high amount of nan values in column:
  data.drop(
    ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'verification_status_joint',
     'desc'], axis=1, inplace=True)

  # Dropping because of high cardinality
  data.drop(['emp_title', 'title', 'batch_enrolled'], axis=1, inplace=True)

  #Dropping because it wasn't found any pattern between the state and the loan status:
  data.drop(['zip_code', 'addr_state', 'sub_grade'], axis = 1, inplace = True)

  # Categorical features:
  data['term'] = data['term'].apply(lambda x: int(x.split(' ')[0]))

  data['initial_list_status'] = data['initial_list_status'].apply(lambda x: 0 if x == 'f' else 1)

  data['emp_length'].fillna('Missing', inplace=True)
  data['emp_length'] = data['emp_length'].str.replace('(year.*)', '', regex=True)
  data['emp_length'] = data['emp_length'].str.replace(' ', '')
  data['emp_length'] = data['emp_length'].str.replace('<1', '0')

  categorical_variables = ['term', 'emp_length', 'initial_list_status']

  return data[categorical_variables]


#Label encodes the categorical features and generates a dictionary with a LabelEncoder for each feature to be used to prepare the testing data.
def labelencode_categorical_data(data):

  categorical_variables = data.columns

  categorical_label_encoders = {x:LabelEncoder() for x in categorical_variables}

  for variable in categorical_variables:
    categorical_label_encoders[variable].fit(data[variable])
    data[variable] = categorical_label_encoders[variable].transform(data[variable])

  return data ,categorical_label_encoders




def clean_numerical_data(data):
  numerical_variables = ['last_week_pay', 'loan_amnt', 'funded_amnt', 'dti', 'int_rate', 'annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'total_rec_int', 'pub_rec', 'last_week_pay', 'revol_util', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

  #Determines how many weeks since last payment for the loan.
  def last_week_pay_parse(x):
    x = x.replace('th week', '')
    if x == 'NA':
      return 0
    else:
      return int(x)

  data['last_week_pay'] = data['last_week_pay'].apply(lambda x: last_week_pay_parse(x))

  #It was found that most of the numerical features had a consistent number of outliers and also, null values.
  #To fill these values, the mean of the values between the non-outliers values was calculated.
  for variable in numerical_variables:
    q1 = data[variable].quantile(0.25)
    q3 = data[variable].quantile(0.75)

    if q1 == 0 and q3 == 0:
      data[variable].fillna(0.0, inplace=True)
    else:

      iqr = q3 - q1
      limit1 = q1 - 1.5 * iqr
      limit2 = q3 + 1.5 * iqr

      #mean is extracted for that variable and then, used to fill the nan.
      mean = np.nanmean(
        np.extract(
          (data[variable] >= limit1) &
          (data[variable] <= limit2),
          arr = data[variable]
        )
      )
      data[variable].fillna(mean, inplace=True)

  return data[numerical_variables]


###It was observed that it was possible to reduce the number of features by combining them to obtain ratios that will also add insights. ###
def feature_engineering(data):

  data['inc_loan_ratio'] = (data['annual_inc'] / data['loan_amnt'].apply(lambda x: 1 if x == 0 else x)).astype(np.float16)
  data['borrow_loan_ratio'] = (data['total_rev_hi_lim']/data['loan_amnt'].apply(lambda x: 1 if x == 0 else x)).astype(np.float16)
  data['is_delinquent'] = data['acc_now_delinq'].apply(lambda x: 0 if x == 0 else 1)
  data['is_delinquent'] = data['is_delinquent'].astype(np.int8)
  data['funded_ratio'] = (data['loan_amnt'] / data['funded_amnt']).astype(np.float16)
  data.drop(['annual_inc', 'loan_amnt', 'total_rev_hi_lim', 'loan_amnt', 'acc_now_delinq'], axis = 1, inplace = True)

  return data



###The numerical features were scaling between 0 and 1 to improve the model's efficiency of prediction.
# This returns the numerical variables scaled and the scaler that will be later used for the testing data.###
def min_max_scaling(data):
  min_max_scaler = MinMaxScaler()
  min_max_scaler.fit(data)
  scaled_data = min_max_scaler.transform(data)
  return pd.DataFrame(scaled_data, columns = data.columns), min_max_scaler


train_data = pd.read_csv('train_indessa.csv')

train_data.drop('member_id', axis = 1, inplace = True)

#Categorical features are processed and cleaned:
clean_categorical_data = clean_categorical_data(train_data)
encoded_categorical_data, label_encoders = labelencode_categorical_data(clean_categorical_data)
resulting_categorical_columns = list(encoded_categorical_data.columns)

#Numerical features are processed and cleaned:
clean_numerical_data = clean_numerical_data(train_data)
featured_engineered_data = feature_engineering(clean_numerical_data)
scaled_numerical_data, min_max_scaler = min_max_scaling(featured_engineered_data)

#Data is finally combined :
data = pd.concat([encoded_categorical_data, scaled_numerical_data], axis=1)
data.columns = resulting_categorical_columns + list(scaled_numerical_data.columns)

data['loan_status'] = train_data['loan_status'].astype(np.int8)


#It was discovered that the dataset had a huge imbalance. Given that there are plenty of rows for each case,
#rows with the most populated loan status value were randomly eliminated until there were exactly the same of number for both categories.
rus = RandomUnderSampler(random_state=42)
y = data['loan_status']
data.drop('loan_status', axis = 1, inplace = True)
X, y = rus.fit_resample(data, y)

data = X
data['loan_status'] = y

#Data is stored to train the model in train_model.py
data.to_csv('balanced_training_data.csv', index = False)
