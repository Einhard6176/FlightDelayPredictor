import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats

from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import warnings
warnings.filterwarnings("ignore")

from functions import *


# Reading data - Specify where the data is coming from
print('Loading data...')
data = pd.read_csv('random3000000.csv')
print('Flight data loaded!')
passengers = pd.read_csv('passengers.csv', sep=',')
print('Passenger data loaded!')
testdata = pd.read_csv('flight_test_with_30d.csv')
print('Preparing testdata...')
testdata = disperseDate(testdata)
testdata = getDepArrHour(testdata)
testdata = encodeCarrier(testdata)
print('Testdata loaded!')

# Remove outliers (keep only rows where z-score is less than 3)
data['arr_delay_z'] = data['arr_delay']
data['arr_delay_z'] = data['arr_delay_z'].fillna(0)
z = pd.DataFrame(np.abs(stats.zscore(data['arr_delay_z'])))
z = z.rename(columns={0:'abs_zscore'})
data = data.join(z)
data = data[data['abs_zscore'] < 3]
data.drop(['arr_delay_z','abs_zscore'], axis = 1 ,inplace=True)



######################################################################
# Binnary Classification - cancellation prediction
cancellations_query = ['fl_date',
                       'dep_time',
                       'op_unique_carrier',
                       'origin_airport_id',
                       'dest_airport_id',
                       'taxi_out',
                       'arr_delay',
                       'taxi_in',
                       'distance',
                       'crs_elapsed_time',
                       'crs_dep_time',
                       'crs_arr_time',
                       'dep_delay',
                       'carrier_delay',
                       'weather_delay',
                       'nas_delay',
                       'security_delay',
                       'late_aircraft_delay',
                       'origin_city_name',
                       'dest_city_name',
                       'cancelled']
print('Creating cancellations df...')
cancellations_data = data[cancellations_query]

# Adding features
print('Adding statistics to cancellation df...')
cancellations_data = getStatistics(cancellations_data)
print('Filling statistics...')
cancellations_data = fillStatistics(cancellations_data)
print('Prepping and encoding data...')
cancellations_data = binnaryPrep(cancellations_data)
cancellations_data = encodeCarrier(cancellations_data)
cancellations_data = disperseDate(cancellations_data)
cancellations_data = cancellations_data.drop('fl_date', axis=1)

# Creating test/train split
print('Creating test/train split...')
X_cancel = cancellations_data[['year',
                               'month',
                               'day',
                               'origin_airport_id',
                               'dep_hr',
                               'arr_hr',
                               'carrier',
                               'crs_elapsed_time',
                               'distance',
                               'dest_airport_id',
                               'airport_30D_avg_dep_delay',
                               'airport_30D_avg_arr_delay',
                               'airport_30D_min_dep_delay',
                               'airport_30D_max_dep_delay',
                               'airport_30D_min_arr_delay',
                               'airport_30D_max_arr_delay',
                               'carrier_30D_avg_dep_delay',
                               'carrier_30D_avg_arr_delay',
                               'carrier_30D_min_dep_delay',
                               'carrier_30D_max_dep_delay',
                               'carrier_30D_min_arr_delay',
                               'carrier_30D_max_arr_delay',
                               'airport_30D_avg_taxi_out',
                               'airport_30D_avg_taxi_in']]

y_cancel = cancellations_data['cancelled']


# Standarizing and fitting
print('Standarizing...')
scaler = StandardScaler()
X_cancel_std = pd.DataFrame(data = scaler.fit_transform(X_cancel),columns=X_cancel.columns)

# Oversampling
print('Splitting and oversampling...')
X_cancel_train, X_cancel_test, y_cancel_train, y_cancel_test = train_test_split(X_cancel_std, y_cancel, test_size=0.3)
sm = SMOTE()
X_train_over,y_train_over = sm.fit_resample(X_cancel_train.values,y_cancel_train.values)

# FIT MODEL AND RETRIEVE Y_PRED
print('Fitting model...')
knnBinnary = KNeighborsClassifier(n_neighbors = 2)
knnBinnary.fit(X_train_over, y_train_over)
print('Success!')
y_predBinary = knnBinnary.predict(testdata[X_cancel.columns.tolist()])



######################################################################
# Multiclass Classification - predicting delay type

delayType_query = ['fl_date',
                   'dep_time',
                   'op_unique_carrier',
                   'origin_airport_id',
                   'dest_airport_id',
                   'taxi_out',
                   'arr_delay',
                   'taxi_in',
                   'distance',
                   'crs_elapsed_time',
                   'crs_dep_time',
                   'crs_arr_time',
                   'dep_delay',
                   'carrier_delay',
                   'weather_delay',
                   'nas_delay',
                   'security_delay',
                   'late_aircraft_delay',
                   'origin_city_name',
                   'dest_city_name']
print('Creating delays df...')
delayType_data = data[delayType_query]

print('Adding statistics to delays df...')
delayType_data = getStatistics(delayType_data)

print('Prepping and cleaning delays data...')
delayType_data = fillStatistics(delayType_data)
delayType_data = multiclassPrep(delayType_data)
delayType_data = encodeCarrier(delayType_data)

print('Creating test/train split...')
X_delay_type = delayType_data[['fl_date',
                               'origin_airport_id',
                               'dep_hr',
                               'arr_hr','carrier',
                               'crs_elapsed_time',
                               'distance',
                               'dest_airport_id',
                               'distance',
                               'airport_30D_avg_dep_delay',
                               'airport_30D_avg_arr_delay',
                               'airport_30D_min_arr_delay',
                               'airport_30D_max_arr_delay',
                               'airport_30D_min_dep_delay',
                               'airport_30D_max_dep_delay',
                               'carrier_30D_avg_dep_delay',
                               'carrier_30D_avg_arr_delay',
                               'carrier_30D_min_dep_delay',
                               'carrier_30D_max_dep_delay',
                               'carrier_30D_min_arr_delay',
                               'carrier_30D_max_arr_delay',
                               'airport_30D_avg_taxi_out',
                               'airport_30D_avg_taxi_in',]]

X_delay_type = disperseDate(X_delay_type).drop('fl_date', axis=1)


y_delay_type = delayType_data[['nas_delay',
                               'late_aircraft_delay',
                               'carrier_delay',
                               'weather_delay',
                               'security_delay']]

y_delay_type = delayTypeEncoder(y_delay_type)
y_delay_type=y_delay_type.replace({'na':0,
                                   'late_aircraft_delay':1,
                                   'nas_delay':2,
                                   'carrier_delay':3,
                                   'weather_delay':4,
                                   'security_delay':5
})

y_delay_type = y_delay_type['delay_type']


# FIT MODEL AND RETRIEVE Y_PRED
X_train_delay, X_test_delay, y_train_delay, y_test_delay = train_test_split(X_delay_type, y_delay_type, test_size=0.3)

print('Fitting model...')
knnMulticlass = KNeighborsClassifier(n_neighbors=4)
knnMulticlass.fit(X_train_delay,y_train_delay)
y_predMulticlass = knnMulticlass.predict(testdata[X_delay_type.columns.tolist()])
print('Sucess!')


######################################################################
# Arrival Delay predictions

flight_query = ['fl_date',
                 'dep_delay',
                 'op_unique_carrier',
                 'origin_airport_id',
                 'dest_airport_id',
                 'taxi_out',
                 'arr_delay',
                 'taxi_in',
                 'distance',
                 'air_time',
                 'carrier_delay',
                 'weather_delay',
                 'security_delay',
                 'crs_dep_time',
                 'crs_arr_time',
                 'nas_delay',
                 'late_aircraft_delay',
                 'origin_city_name',
                 'dest_city_name',
                 'cancelled']

print('Creating cancellations df...')
flight_data = data[flight_query]

print('Adding cancellations prediction...')
testdata['cancelled'] = y_predBinary

print('Computing statistics...')
flight_data = getStatistics(flight_data)
flight_data = makeDepHrBins(flight_data)
flight_data.fl_date = pd.to_datetime(flight_data.fl_date)
flight_data['day_of_week'] = flight_data.fl_date.dt.dayofweek


# Creating regional classification
print('Creating regional classifications for train data')
flight_data = getState(flight_data)
flight_data = getRegion(flight_data)
df_region_dummies = pd.get_dummies(flight_data['region'], prefix='reg')
flight_data = flight_data.join(df_region_dummies)

print('Creating regional classifications for testdata')
testdata = getState(testdata)
testdata = getRegion(testdata)
test_region_dummies = pd.get_dummies(testdata['region'], prefix='reg')
testdata = testdata.join(test_region_dummies)



# Encoding delay types for testdata
print('Encoding delay types...')
testdata['predicted_delay_types'] = y_predMulticlass
test_delay_dummies = pd.get_dummies(testdata['predicted_delay_types'], prefix='d_type')
testdata = testdata.join(test_delay_dummies)
# Encoding delay types for flight data
df_delays = flight_data[['nas_delay','late_aircraft_delay','carrier_delay','weather_delay','security_delay']]
df_delays_post = delayTypeEncoder(df_delays)
df_delays_post = df_delays_post.replace({'na':0,
                                   'late_aircraft_delay':1,
                                   'nas_delay':2,
                                   'carrier_delay':3,
                                   'weather_delay':4,
                                   'security_delay':5
})
flight_delay_dummies = pd.get_dummies(df_delays_post['delay_type'], prefix='d_type')
flight_data = flight_data.join(flight_delay_dummies)


print('Cleaning and prepping data...')

# Keeping numerical columns
flight_data_numeric = flight_data.drop(['fl_date',
                                        'origin_airport_id',
                                        'dep_hr',
                                        'op_unique_carrier',
                                        'dep_delay',
                                        'dest_airport_id',
                                        'taxi_out',
                                        'taxi_in',
                                        'distance',
                                        'air_time',
                                        'carrier_delay',
                                        'weather_delay',
                                        'security_delay',
                                        'crs_dep_time',
                                        'crs_arr_time',
                                        'nas_delay',
                                        'late_aircraft_delay',
                                        'arr_hr',
                                        'origin_city_name',
                                        'dest_city_name',
                                        'region',
                                        'dep_state',
                                        'dep_hr_bin',
                                        'day_of_week',
                                        'bin_mean_delay'
                                        ], axis = 1)



# Filling NaNs or dropping them
flight_data_numeric['airport_30D_avg_taxi_in'] = flight_data_numeric['airport_30D_avg_taxi_in'].fillna(flight_data['taxi_in'].mean())
flight_data_numeric = flight_data_numeric.dropna()

# Fit and model

X_flight_data = flight_data_numeric.drop(['arr_delay'], axis = 1)
y_flight_data = flight_data_numeric['arr_delay']
flight_data_numeric = flight_data_numeric.drop(['arr_delay'], axis = 1)

print('Scaling data...')
scaler = StandardScaler()
X_flight_data_std = scaler.fit_transform(X_flight_data)
X_flight_data_std = pd.DataFrame(data=X_flight_data_std, columns=X_flight_data.columns)

print('Splitting data')
X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(X_flight_data_std, y_flight_data, test_size=0.3)

print('Scale Test Data...')
testdata_std = scaler.fit_transform(testdata[flight_data_numeric.columns.tolist()])
testdata_std = pd.DataFrame(data=testdata_std, columns=testdata[flight_data_numeric.columns.tolist()].columns)

# Run the different predictive models
print('Starting Gradient Boost Regressor...')
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train_arr, y_train_arr)
y_predgbr = gbr.predict(X_test_arr)
y_pred_test_gbr = gbr.predict(testdata_std)
score_gbr = r2_score(y_test_arr, y_predgbr)
print('R2 = ', score_gbr)

print('Starting Linear Regression (Degree = 1)...')
degree = 1
polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg.fit(X_train_arr,y_train_arr)
y_predArr = polyreg.predict(testdata_std)
y_pred_split = polyreg.predict(X_test_arr)
print('Sucess!')
r2_deg1 = r2_score(y_test_arr, y_pred_split)
mse_deg1 = mean_squared_error(y_test_arr, y_pred_split)
rmse_deg1 = np.sqrt(mse_deg1)
mae_deg1 = mean_absolute_error(y_test_arr, y_pred_split)
print('R2 = ' + str(r2_deg1))

print('Starting Ridge regression...')
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0.3)
ridgereg.fit(X_train_arr, y_train_arr)
y_pred_test_ridge = ridgereg.predict(testdata_std)
y_predridge = ridgereg.predict(X_test_arr)
score_ridge = r2_score(y_test_arr, y_predridge)
print("R2 = ", score_ridge)

print('Starting Lasso regression...')
from sklearn.linear_model import Lasso
lassoreg = Lasso(alpha=0.3)
lassoreg.fit(X_train_arr, y_train_arr)
y_pred_test_lasso = lassoreg.predict(testdata_std)
y_predlasso = lassoreg.predict(X_test_arr)
score_lasso = r2_score(y_test_arr, y_predlasso)
print("R2 = ", score_lasso)

print('Starting Random Forest Regressor...')
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train_arr, y_train_arr)
y_predrfr = rfr.predict(X_test_arr)
y_pred_test_rfr = rfr.predict(testdata_std)
score_rfr = r2_score(y_test_arr, y_predrfr)
print('R2 = ', score_rfr)
