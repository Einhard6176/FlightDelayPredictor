import pandas as pd
import numpy as np
import psycopg2
import scipy.stats as stats
from sklearn import preprocessing

def SQLquery(query):
    host = 'mid-term-project.ca2jkepgjpne.us-east-2.rds.amazonaws.com'
    port = "5432"
    user = 'lhl_student'
    pwd = 'lhl_student'
    database = 'mid_term_project'
    conn = psycopg2.connect(user = user,
                              password = pwd,
                              host = host,
                              port = port,
                              database = database)
    return pd.read_sql(query, conn)


def nanPercent(series):
    nan_percentage = round(float(series.isna().sum()/series.count()*100))
    return print(f'{nan_percentage}% of values are NaN')

def trafficPercentByState(df):
    '''
    Returns dfframe with total aircraft movements and percentage of totals
    Note that only works when all states are present - needs a sufficiently large sample
    '''
    df['dest_state'] = df.dest_city_name.str[-2:]
    df['origin_state'] = df.origin_city_name.str[-2:]
    movements = df[['origin_state', 'fl_date']].groupby('origin_state').count()['fl_date'] + df[['dest_state', 'fl_date']].groupby('dest_state').count()['fl_date']
    movements = pd.dfFrame(movements)
    movements['percentage_of_total'] = movements.fl_date/movements.fl_date.sum()*100
    return movements

def cumulativePercentage(df, threshold):
    '''
    Returns airports that cunmulatively consist of threshold percent of total traffic
    '''
    traffic = traffic_percent_by_state(df).sort_values(by='percentage_of_total',   ascending=False)
    return traffic[traffic.cumsum().percentage_of_total <= threshold].rename({'fl_date':'total_movements'}, axis=1)

def calculateSpeed(df):
    '''
    Returns column with speed values in MPH
    '''
    df['speed'] = df.distance/df.crs_elapsed_time*60
    return df

def otp(df):
    '''
    Returns On-Time-Performance binary, where 0 = on-time or early departure, else 1
    '''
    df = calculateSpeed(df)
    df['OTP'] = np.where(df.dep_delay <= 0, 0,1)
    return df

def aircraftSpeedTTest(df):
    '''
    Compares and checks whether the distribution of aircraft speeds is the same with a departure delay or not using a T-Test.
    '''
    df = otp(df)
    stat, p_value = stats.ttest_ind(df[df.OTP.isin([0])].speed,df[df.OTP.isin([1])].speed, nan_policy='omit')
    if p_value < 0.05:
        print(f'Hypothesis rejected; aircraft speeds are not equal with delays present.')
        print(f'P-value: {p_value}')
    else:
        print(f'Delays have no effect on aircraft speed')

def getMonth(df):
    df.fl_date = pd.to_datetime(df.fl_date)
    df.fl_date = pd.DatetimeIndex(df.fl_date).month
    return df.rename({'fl_date':'month'}, axis=1)

def getDistance(df):
    distance = df.groupby(['op_unique_carrier','month',]).sum().drop('dep_delay', axis=1)
    return distance

def getHour(time_num):
    return int(time_num/100)

def getDepArrHour(df):
    df['dep_hr'] = df['crs_dep_time'].apply(getHour)
    df['arr_hr'] = df['crs_arr_time'].apply(getHour)

    return df


def hotEncodeCarrier(df):
    carrier_id={}
    key=set(df['op_unique_carrier'])
    #values=list(range(len(key)))
    i=1
    for k in key:
        carrier_id[k]=i
        i=i+1
    df['carrier_id']=df['op_unique_carrier'].replace(to_replace=carrier_id)
    return df

def getDelaySum(df):
    '''
    Adds all specific type delays
    '''
    df['total_delays'] = df.carrier_delay \
                       + df.weather_delay \
                       + df.nas_delay \
                       + df.security_delay \
                       + df.late_aircraft_delay
    df['total_delays'] = df['total_delays'].fillna(0)
    return df

def getState(df):
    df['dep_state'] = df.origin_city_name.str[-2:]
    return df

def getRegion(df):
    # https://www2.census.gov/geo/pdfs/maps-df/maps/reference/us_regdiv.pdf

    pacific = ['WA', 'OR', 'CA', 'AK', 'HI']
    mountain = ['MT', 'ID', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM']
    westNorthCentral = ['ND', 'SD', 'NE',' KS','MN','IA','MO']
    westSouthCentral = ['OK', 'TX','AR','LA']
    eastNorthCentral = ['WI','IL','MI','IN','OH']
    eastSouthCentral = ['KY','TN','MS','AL']
    middleAtlantic = ['NY','PA','NJ']
    newEngland = ['ME','VT','NH','MA','CT','RI']
    southAtlantic = ['WV','MD','DE','DC','VA','NC','SC','GA','FL']

    df['region'] = 'N/A'

    df.region.where(~df.dep_state.isin(pacific), 'pacific', inplace=True)
    df.region.where(~df.dep_state.isin(mountain), 'mountain', inplace=True)
    df.region.where(~df.dep_state.isin(westNorthCentral), 'westNorthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(westSouthCentral), 'westSouthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(eastNorthCentral), 'eastNorthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(eastSouthCentral), 'eastSouthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(middleAtlantic), 'middleAtlantic', inplace=True)
    df.region.where(~df.dep_state.isin(newEngland), 'newEngland', inplace=True)
    df.region.where(~df.dep_state.isin(southAtlantic), 'southAtlantic', inplace=True)
    return df

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

def listDiff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

def disperseDate(df):
    '''
    Separates single datetime columns into three with year, month, day
    '''
    df.fl_date = pd.to_datetime(df.fl_date)
    df['year'] = df.fl_date.dt.year.astype(int)
    df['month'] = df.fl_date.dt.month.astype(int)
    df['day'] =  df.fl_date.dt.day.astype(int)
    return df

def makeDepHrBins(df):
    '''
    Creates bins for distinct traffic pattern periods. Must be run *after* getDepArrHour or getStatistics
    '''
    bins = [2,6,10,14,18,22]
    labels = ['morning', 'noon','afternoon','evening','night']
    df['dep_hr_bin'] = pd.cut(pd.to_datetime(df.dep_hr, format='%H').dt.hour, bins, labels=labels, right=False)
    df.dep_hr_bin = df.dep_hr_bin.astype('str').replace('nan', 'late_night')
    df = df.set_index('dep_hr_bin')
    df['bin_mean_delay'] = df.groupby('dep_hr_bin').mean().arr_delay
    df = df.reset_index()
    return df

def getFreightPaxProba(flights, passengers):
    empty_flights = passengers.loc[(passengers['seats'] == 0.0) & (passengers['passengers'] == 0.0)].sort_values(by='freight')

    passenger_flights = passengers.drop(empty_flights.index)

    num_empty_per_carrier = empty_flights[['unique_carrier','departures_scheduled']].groupby('unique_carrier').sum().sort_values(by='departures_scheduled',ascending=False)

    num_pass_per_carrier = passenger_flights[['unique_carrier','departures_scheduled']].groupby('unique_carrier').sum().sort_values(by='departures_scheduled',ascending=False)

    num_empty_per_carrier = num_empty_per_carrier.rename(columns={'departures_scheduled':'total_cargo_departures'})

    num_pass_per_carrier = num_pass_per_carrier.rename(columns={'departures_scheduled':'total_passenger_departures'})

    merged_dep_counts = num_pass_per_carrier.merge(num_empty_per_carrier, how='outer', on=['unique_carrier'])

    merged_dep_counts = merged_dep_counts.fillna(0.0)

    merged_dep_counts['total_departures'] = merged_dep_counts['total_passenger_departures'] + merged_dep_counts['total_cargo_departures']

    merged_dep_counts['pax_proba'] = (merged_dep_counts['total_passenger_departures']/merged_dep_counts['total_departures']) * 100
    merged_dep_counts['cargo_proba'] = (merged_dep_counts['total_cargo_departures']/merged_dep_counts['total_departures']) * 100

    to_drop = merged_dep_counts.loc[(merged_dep_counts['total_departures'] == 0)]
    merged_dep_counts = merged_dep_counts.drop(to_drop.index)

    return merged_dep_counts

#######
# Modelling

##########################################################################################################################################
##########################################################################################################################################

# Airport Statistics

def getAirportDelayStatistics(df, arr_or_dep, window=30, offset=7):
    '''
    Appends column to dfframe with rolling average arrival or delay over specified days with specified offset( deault is 7 day offset, 30 day window)

    Note, Airportdf needs to be multi-indexed by fl_date and then origin_airport_id first.    '''


    df_means = df.groupby(['origin_airport_id','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['origin_airport_id'])[arr_or_dep].rolling(window, min_periods=1).mean().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'origin_airport_id']).reset_index()

    df_means = df_means.set_index(['fl_date', 'origin_airport_id'])['rolling_value']
    return df_means

def getAirportMinStatistics(df, arr_or_dep,window=30, offset=7):
    '''
    Appends column to dfframe with rolling min arrival or delay over 30 days with specified offset ( deault is 7 day offset, 30 day window)

    Note, df needs to be multi-indexed by fl_date and then origin_airport_id first.
    '''
    df_means = df.groupby(['origin_airport_id','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['origin_airport_id'])[arr_or_dep].rolling(window, min_periods=1).min().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'origin_airport_id']).reset_index()

    df_means = df_means.set_index(['fl_date', 'origin_airport_id'])['rolling_value']
    return df_means

def getAirportMaxStatistics(df, arr_or_dep,window=30, offset=7):
    '''
    Appends column to dfframe with rolling min arrival or delay over 30 days with specified offset ( deault is 7 day offset, 30 day window)

    Note, df needs to be multi-indexed by fl_date and then origin_airport_id first.

    '''
    df_means = df.groupby(['origin_airport_id','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['origin_airport_id'])[arr_or_dep].rolling(window, min_periods=1).max().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'origin_airport_id']).reset_index()

    df_means = df_means.set_index(['fl_date', 'origin_airport_id'])['rolling_value']
    return df_means

def getTaxiTimeStatistics(df, arr_or_dep, window=30, offset=7):
    '''
    Appends column to dfframe with rolling average arrival or delay over specified days with specified offset( deault is 7 day offset, 30 day window)
    Note, df needs to be multi-indexed by fl_date, origin_airport_id, and then hr first.
    '''

    df_means = df.groupby(['origin_airport_id',arr_or_dep,'fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['origin_airport_id',arr_or_dep])['taxi_out'].rolling(window, min_periods=1).mean().shift(offset).reset_index().rename({'taxi_out':'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'origin_airport_id',arr_or_dep]).reset_index()

    df_means = df_means.set_index(['fl_date', 'origin_airport_id',arr_or_dep])['rolling_value']
    return df_means

##########################################################################################################################################
##########################################################################################################################################

# Carrier Statistics

def getCarrierDelayStatistics(df, arr_or_dep, window=30, offset=7):
    '''
    Appends column to dfframe with rolling average arrival or delay over specified days with specified offset( deault is 7 day offset, 30 day window)
    '''
    df_means = df.groupby(['op_unique_carrier','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['op_unique_carrier'])[arr_or_dep].rolling(window, min_periods=1).mean().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'op_unique_carrier']).reset_index()

    df_means = df_means.set_index(['fl_date', 'op_unique_carrier'])['rolling_value']
    return df_means

def getCarrierMinStatistics(df, arr_or_dep,window=30, offset=7):
    '''
    Appends column to dfframe with rolling min arrival or delay over 30 days with specified offset ( deault is 7 day offset, 30 day window)

    Note, df needs to be multi-indexed by fl_date and then op_unique_carrier first.
    '''
    df_means = df.groupby(['op_unique_carrier','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['op_unique_carrier'])[arr_or_dep].rolling(window, min_periods=1).min().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'op_unique_carrier']).reset_index()

    df_means = df_means.set_index(['fl_date', 'op_unique_carrier'])['rolling_value']
    return df_means

def getCarrierMaxStatistics(df, arr_or_dep,window=30, offset=7):
    '''
    Appends column to dfframe with rolling min arrival or delay over 30 days with specified offset ( deault is 7 day offset, 30 day window)

    Note, df needs to be multi-indexed by fl_date and then op_unique_carrier first.
    '''
    df_means = df.groupby(['op_unique_carrier','fl_date']).mean().reset_index().set_index('fl_date').sort_values(by='fl_date').groupby(['op_unique_carrier'])[arr_or_dep].rolling(window, min_periods=1).max().shift(offset).reset_index().rename({arr_or_dep:'rolling_value'}, axis=1).set_index('fl_date').sort_values(by=['fl_date', 'op_unique_carrier']).reset_index()

    df_means = df_means.set_index(['fl_date', 'op_unique_carrier'])['rolling_value']
    return df_means

######################################################################################################################################################################################

def getStatistics(df, window=30, offset=7):
    df = df.set_index(['fl_date', 'origin_airport_id'])

    # Creates 30-day rolling average, min, and max delay for each origin and destination airport
    df['airport_30D_avg_dep_delay'], df['airport_30D_avg_arr_delay'] = getAirportDelayStatistics(df, 'dep_delay',window, offset), getAirportDelayStatistics(df, 'arr_delay',window, offset)
    df['airport_30D_min_dep_delay'], df['airport_30D_max_dep_delay'] = getAirportMinStatistics(df, 'dep_delay',window, offset), getAirportMaxStatistics(df, 'dep_delay',window, offset)
    df['airport_30D_min_arr_delay'], df['airport_30D_max_arr_delay'] = getAirportMinStatistics(df, 'arr_delay',window, offset), getAirportMaxStatistics(df, 'arr_delay',window, offset)

    # Resetting index for next operations
    df = df.reset_index()
    df = df.set_index(['fl_date', 'op_unique_carrier'])

    # Creates 30-day rolling average, min, and max delay for each carrier
    df['carrier_30D_avg_dep_delay'], df['carrier_30D_avg_arr_delay'] = getCarrierDelayStatistics(df, 'dep_delay',window, offset), getCarrierDelayStatistics(df, 'arr_delay',window, offset)
    df['carrier_30D_min_dep_delay'], df['carrier_30D_max_dep_delay'] = getCarrierMinStatistics(df, 'dep_delay',window, offset), getCarrierMaxStatistics(df, 'dep_delay',window, offset)
    df['carrier_30D_min_arr_delay'], df['carrier_30D_max_arr_delay'] = getCarrierMinStatistics(df, 'arr_delay',window, offset), getCarrierMaxStatistics(df, 'arr_delay',window, offset)

    # Creates 30-day rolling average for taxitimes by hour
    df = getDepArrHour(df)
    df = df.reset_index()
    df=df.set_index(['fl_date', 'origin_airport_id','dep_hr'])
    df['airport_30D_avg_taxi_out'] = getTaxiTimeStatistics(df, 'dep_hr')
    df['airport_30D_avg_taxi_in'] = getTaxiTimeStatistics(df, 'arr_hr')


    df = df.reset_index()
    return df

#######################################################
# Metrics

def print_results(headline, true_value, pred):
    print(headline)
    print("Accuracy: {}".format(metrics.accuracy_score(true_value, pred)))
    print("Precision: {}".format(metrics.precision_score(true_value, pred)))
    print("Recall: {}".format(metrics.recall_score(true_value, pred)))
    print("F1: {}".format(metrics.f1_score(true_value, pred)))
    print("F2: {}".format(metrics.fbeta_score(true_value, pred, beta=2)))




######################################################

# Unfinished or non-functional


def getRegion(df):
    # https://www2.census.gov/geo/pdfs/maps-df/maps/reference/us_regdiv.pdf
    pacific = ['WA', 'OR', 'CA', 'AK', 'HI']
    mountain = ['MT', 'ID', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM']
    westNorthCentral = ['ND', 'SD', 'NE',' KS','MN','IA','MO']
    westSouthCentral = ['OK', 'TX','AR','LA']
    eastNorthCentral = ['WI','IL','MI','IN','OH']
    eastSouthCentral = ['KY','TN','MS','AL']
    middleAtlantic = ['NY','PA','NJ']
    newEngland = ['ME','VT','NH','MA','CT','RI']
    southAtlantic = ['WV','MD','DE','DC','VA','NC','SC','GA','FL']

    df['region'] = 'N/A'
    df.region.where(~df.dep_state.isin(pacific), 'pacific', inplace=True)
    df.region.where(~df.dep_state.isin(mountain), 'mountain', inplace=True)
    df.region.where(~df.dep_state.isin(westNorthCentral), 'westNorthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(westSouthCentral), 'westSouthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(eastNorthCentral), 'eastNorthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(eastSouthCentral), 'eastSouthCentral', inplace=True)
    df.region.where(~df.dep_state.isin(middleAtlantic), 'middleAtlantic', inplace=True)
    df.region.where(~df.dep_state.isin(newEngland), 'newEngland', inplace=True)
    df.region.where(~df.dep_state.isin(southAtlantic), 'southAtlantic', inplace=True)

    return df

#############################################################
# Cancellations binary classification functions
def fillStatistics(df, val=0):
    # fillna
    df['airport_30D_avg_arr_delay'] = df['airport_30D_avg_arr_delay'].fillna(val)
    df['airport_30D_avg_dep_delay'] = df['airport_30D_avg_dep_delay'].fillna(val)
    df['airport_30D_min_dep_delay'] = df['airport_30D_min_dep_delay'].fillna(val)
    df['airport_30D_max_dep_delay'] = df['airport_30D_max_dep_delay'].fillna(val)
    df['airport_30D_min_arr_delay'] = df['airport_30D_min_arr_delay'].fillna(val)
    df['airport_30D_max_arr_delay'] = df['airport_30D_max_arr_delay'].fillna(val)
    df['carrier_30D_avg_dep_delay'] = df['airport_30D_avg_dep_delay'].fillna(val)
    df['carrier_30D_avg_arr_delay'] = df['carrier_30D_avg_arr_delay'].fillna(val)
    df['carrier_30D_min_dep_delay'] = df['carrier_30D_min_dep_delay'].fillna(val)
    df['carrier_30D_max_dep_delay'] = df['carrier_30D_max_dep_delay'].fillna(val)
    df['airport_30D_max_dep_delay'] = df['airport_30D_max_dep_delay'].fillna(val)
    df['carrier_30D_min_arr_delay'] = df['carrier_30D_min_arr_delay'].fillna(val)
    df['carrier_30D_max_arr_delay'] = df['carrier_30D_max_arr_delay'].fillna(val)
    df['airport_30D_avg_taxi_out'] = df['airport_30D_avg_taxi_out'].fillna(val)
    df['airport_30D_avg_taxi_in'] = df['airport_30D_avg_taxi_in'].fillna(val)

    return df


def multiclassPrep(df):
    df = df[~(df['crs_elapsed_time'].isnull())]
    df = df.reset_index(drop=True)
    return df


def binnaryPrep(df):
    df['crs_elapsed_time'] = df['crs_elapsed_time'].fillna((df['crs_elapsed_time']).mean())
    df = df.reset_index(drop = True)
    return df


def encodeCarrier(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['op_unique_carrier'])
    df['carrier']=le.transform(df['op_unique_carrier'])
    return df


def delayTypeEncoder(df):
    delay_type=[]
    for i in df.index:
        if df.loc[i].isnull().sum()==5:
            delay_type.append('na')
        else:
            ini=0
            for delay in df.loc[i].index:
                if float(df.loc[i].loc[delay])>ini: #Jimi's correction
                    ini=df.loc[i].loc[delay]
                    d=delay
                else:
                    pass
            delay_type.append(d)
    return pd.DataFrame({'delay_type':delay_type})


#
#def delayTypeEncoder(y):
#    delay_type = []
#    for i in y.index:
#        if y.loc[i].isnull().sum() == 5:
#            delay_type.append('n/a')
#        else:
#            ini = 0
#            for delay in y.loc[i].index:
#                if y.loc[i].loc[delay] > ini:
#                    ini = y.loc[i].loc[delay]
#                    d = delay
#                else:
#                    pass
#            delay_type.append(d)
#
#    return pd.DataFrame(delay_type,columns=['delay_type']).reset_index(drop=True)
#

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
