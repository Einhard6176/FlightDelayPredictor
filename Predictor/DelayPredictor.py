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
data = pd.read_csv('random1000000.csv')
#.drop('Unnamed: 0', axis=1)
print('Flight data loaded!')
#df_weather = pd.read_csv('/media/einhard/Seagate Expansion Drive/lighthouse/midterm/data/weather_data_complete.csv')
#print('Weather data loaded!')
passengers = pd.read_csv('passengers.csv', sep=',')
print('Passenger data loaded!')
testdata = pd.read_csv('flight_test_with_30d.csv')
#.drop('Unnamed: 0', axis=1)
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


#def getWeather(date,city):
#    weather=0
#    try:
#        string=str(df_weather[df_weather['Date']==date][city].values)
#    except:
#        city=city[-2:]
#        for i in df_weather.columns:
#            if city in i:
#                city=i
#                break
#        string=str(df_weather[df_weather['Date']==date][city].values)
#
#
#    if 'Clear' in string:
#        weather=0
#    elif 'Rain' in string:
#        weather=2
#    elif 'Snow' in string:
#        weather=3
#    else:
#        weather=1
#    return weather


######################################################################
# Binnary Classification - cancellation prediction - from Viki's work
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

## Adding weather data
#print('Adding weather data...')
#cancellations_data['weather_code_org'] = cancellations_data['fl_date'].combine(cancellations_data#['origin_city_name'], func=getWeather)
#cancellations_data['weather_code_dest'] = cancellations_data['fl_date'].combine#(cancellations_data['dest_city_name'], func=getWeather)

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
# Multiclass Classification - predicting delay type- from Viki's work

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

#print('Adding weather data...')
#delayType_data['weather_code_org'] = delayType_data['fl_date'].combine(delayType_data#['origin_city_name'], func=getWeather)
#delayType_data['weather_code_dest'] = delayType_data['fl_date'].combine(delayType_data#['dest_city_name'], func=getWeather)

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
y_delay_type=y_delay_type.replace({'na':0, #Jimi's correction
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
df_delays = flight_data[['nas_delay','late_aircraft_delay','carrier_delay','weather_delay','security_delay']] #Jimi added this line
df_delays_post = delayTypeEncoder(df_delays) #Jimi's correction
df_delays_post = df_delays_post.replace({'na':0, #Jimi's correction
                                   'late_aircraft_delay':1,
                                   'nas_delay':2,
                                   'carrier_delay':3,
                                   'weather_delay':4,
                                   'security_delay':5
})
flight_delay_dummies = pd.get_dummies(df_delays_post['delay_type'], prefix='d_type') #Jimi's addition
flight_data = flight_data.join(flight_delay_dummies) #Jimi's addition




## Include weather
#print('Adding weather...')
#flight_data['weather_code_orig'] = flight_data['fl_date'].combine(flight_data['origin_city_name'], #func=getWeather)
#flight_data['weather_code_dest'] = flight_data['fl_date'].combine(flight_data['dest_city_name'], #func=getWeather)

### Encoding weather conditions
#df_o_weath_dummies = pd.get_dummies(flight_data['weather_code_orig'], prefix='o_weath')
#df_d_weath_dummies = pd.get_dummies(flight_data['weather_code_dest'], prefix='d_weath')
#flight_data = flight_data.join(df_o_weath_dummies)
#flight_data = flight_data.join(df_d_weath_dummies)
#print('Sucess!')


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
                                        #'delay_type', Jimi: Remove?
                                        'arr_hr',
                                        #'weather_code_orig', Jimi: Removing weather date for now
                                        #'weather_code_dest', Jimi: Removing weather date for now
                                        'origin_city_name',
                                        'dest_city_name',
                                        'region',
                                        'dep_state',
                                        'dep_hr_bin',
                                        'day_of_week', #Jimi added this line
                                        #'arr_delay',
                                        'bin_mean_delay', #Jimi added this line
                                        'd_type_5' #Jimi: Does not seem to be in testdata
                                        #'predicted_delay_types' Jimi: is ths supposed to be here?
                                        ], axis = 1)



# Filling NaNs or dropping them
flight_data_numeric['airport_30D_avg_taxi_in'] = flight_data_numeric['airport_30D_avg_taxi_in'].fillna(flight_data['taxi_in'].mean())
flight_data_numeric = flight_data_numeric.dropna()

# Fit and model

X_flight_data = flight_data_numeric.drop(['arr_delay'], axis = 1)
y_flight_data = flight_data_numeric['arr_delay']
flight_data_numeric = flight_data_numeric.drop(['arr_delay'], axis = 1) #Jimi addtion

print('Scaling data...')
scaler = StandardScaler()
X_flight_data_std = scaler.fit_transform(X_flight_data)
X_flight_data_std = pd.DataFrame(data=X_flight_data_std, columns=X_flight_data.columns)

print('Splitting data')
X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(X_flight_data_std, y_flight_data, test_size=0.3)

print('Scale Test Data...')
testdata_std = scaler.fit_transform(testdata[flight_data_numeric.columns.tolist()]) #Jimi addition
testdata_std = pd.DataFrame(data=testdata_std, columns=testdata[flight_data_numeric.columns.tolist()].columns) #Jimi addition

# print('Starting Linear Regression (Degree = 1)...')
# degree = 1
# polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
# polyreg.fit(X_train_arr,y_train_arr)
# y_predArr = polyreg.predict(testdata_std)
# y_pred_split = polyreg.predict(X_test_arr) #Jimi added this line
# print('Sucess!')
# r2_deg1 = r2_score(y_test_arr, y_pred_split) #Jimi's correction to show score of y_test and y_pred from the split
# mse_deg1 = mean_squared_error(y_test_arr, y_pred_split) #Jimi's correction to show score of y_test and y_pred from the split
# rmse_deg1 = np.sqrt(mse_deg1) #Jimi's correction to show score of y_test and y_pred from the split
# mae_deg1 = mean_absolute_error(y_test_arr, y_pred_split) #Jimi's correction to show score of y_test and y_pred from the split
# print('MODEL METRICS:')
# print('R2 = ' + str(r2_deg1))
# print('MSE = ' + str(mse_deg1))
# print('RMSE = ' + str(rmse_deg1))
# print('MAE = ' + str(mae_deg1))
#
#
# print('Starting Ridge regression...')  #Jimi addition Thursday morning
# from sklearn.linear_model import Ridge
# ridgereg = Ridge(alpha=0.3) #,normalize=True)
# #poly = PolynomialFeatures(degree = 2)
# #X_ = poly.fit_transform(X_train)
# ridgereg.fit(X_train_arr, y_train_arr)
# #X_ = poly.fit_transform(X_test)
# y_pred_test_ridge = ridgereg.predict(testdata_std)
# y_predridge = ridgereg.predict(X_test_arr)
# score_ridge = r2_score(y_test_arr, y_predridge)
# print("R2 = ", score_ridge)
#
# print('Starting Lasso regression...')  #Jimi addition Thursday morning
# from sklearn.linear_model import Lasso
# lassoreg = Lasso(alpha=0.3) #,normalize=True)
# #poly = PolynomialFeatures(degree = 2)
# #X_ = poly.fit_transform(X_train)
# lassoreg.fit(X_train_arr, y_train_arr)
# #X_ = poly.fit_transform(X_test)
# y_pred_test_lasso = lassoreg.predict(testdata_std)
# y_predlasso = lassoreg.predict(X_test_arr)
# score_lasso = r2_score(y_test_arr, y_predlasso)
# print("R2 = ", score_lasso)
#
# print('Starting Random Forest Regressor...') #Jimi addition Thursday morning
# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor()
# rfr.fit(X_train_arr, y_train_arr)
# y_predrfr = rfr.predict(X_test_arr)
# y_pred_test_rfr = rfr.predict(testdata_std)
# score_rfr = r2_score(y_test_arr, y_predrfr)
# print('R2 = ', score_rfr)

print('Starting Gradient Boost Regressor...') #Jimi addition Thursday morning
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train_arr, y_train_arr)
y_predgbr = gbr.predict(X_test_arr)
y_pred_test_gbr = gbr.predict(testdata_std)
score_gbr = r2_score(y_test_arr, y_predgbr)
print('R2 = ', score_gbr)

# print('Starting Polynomial Regression (Degree = 2)...')
# degree = 2
# polyreg2=make_pipeline(PolynomialFeatures(degree),LinearRegression())
# polyreg2.fit(X_train_arr,y_train_arr)
# y_predArr2 = polyreg2.predict(testdata_std)
# y_pred_split2 = polyreg2.predict(X_test_arr) #Jimi added this line
# print('Sucess!')
# r2_deg2 = r2_score(y_test_arr, y_pred_split2) #Jimi's correction to show score of y_test and y_pred from the split
# mse_deg2 = mean_squared_error(y_test_arr, y_pred_split2) #Jimi's correction to show score of y_test and y_pred from the split
# rmse_deg2 = np.sqrt(mse_deg1) #Jimi's correction to show score of y_test and y_pred from the split
# mae_deg2 = mean_absolute_error(y_test_arr, y_pred_split2) #Jimi's correction to show score of y_test and y_pred from the split
# print('MODEL METRICS:')
# print('R2 = ' + str(r2_deg2))
# print('MSE = ' + str(mse_deg2))
# print('RMSE = ' + str(rmse_deg2))
# print('MAE = ' + str(mae_deg2))

# print('')
# print('Print Scores for each model:')
# print('Linear Regression: ')
# print('R2 = ', str(r2_deg1))
# print('MSE = ' + str(mse_deg1))
# print('RMSE = ' + str(rmse_deg1))
# print('MAE = ' + str(mae_deg1))
# print('mean arr_delay = ' + str(y_predArr.mean()))
# print('')
# print('Ridge Regression: ')
# r2_ridge = r2_score(y_test_arr, y_predridge)
# mse_ridge = mean_squared_error(y_test_arr, y_predridge)
# rmse_ridge = np.sqrt(mse_ridge)
# mae_ridge = mean_absolute_error(y_test_arr, y_predridge)
# print('R2 = ', str(r2_ridge))
# print('MSE = ' + str(mse_ridge))
# print('RMSE = ' + str(rmse_ridge))
# print('MAE = ' + str(mae_ridge))
# print('mean arr_delay = ' + str(y_pred_test_ridge.mean()))
# print('')
# print('Lasso Regression: ')
# r2_lasso = r2_score(y_test_arr, y_predlasso)
# mse_lasso = mean_squared_error(y_test_arr, y_predlasso)
# rmse_lasso = np.sqrt(mse_lasso)
# mae_lasso = mean_absolute_error(y_test_arr, y_predlasso)
# print('R2 = ', str(r2_lasso))
# print('MSE = ' + str(mse_lasso))
# print('RMSE = ' + str(rmse_lasso))
# print('MAE = ' + str(mae_lasso))
# print('mean arr_delay = ' + str(y_pred_test_lasso.mean()))
# print('')
# print('Random Forest: ')
# r2_rfr = r2_score(y_test_arr, y_predrfr)
# mse_rfr = mean_squared_error(y_test_arr, y_predrfr)
# rmse_rfr = np.sqrt(mse_rfr)
# mae_rfr = mean_absolute_error(y_test_arr, y_predrfr)
# print('R2 = ', str(r2_rfr))
# print('MSE = ' + str(mse_rfr))
# print('RMSE = ' + str(rmse_rfr))
# print('MAE = ' + str(mae_rfr))
# print('mean arr_delay = ' + str(y_pred_test_rfr.mean()))
# print('')
# print('Gradient Boost: ')
# r2_gbr = r2_score(y_test_arr, y_predgbr)
# mse_gbr = mean_squared_error(y_test_arr, y_predgbr)
# rmse_gbr = np.sqrt(mse_gbr)
# mae_gbr = mean_absolute_error(y_test_arr, y_predgbr)
# print('R2 = ', str(r2_gbr))
# print('MSE = ' + str(mse_gbr))
# print('RMSE = ' + str(rmse_gbr))
# print('MAE = ' + str(mae_gbr))
# print('mean arr_delay = ' + str(y_pred_test_gbr.mean()))
