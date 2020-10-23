import pandas as pd
import numpy as np
import psycopg2
import scipy.stats as stats

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


def nan_percent(series):
    nan_percentage = round(float(series.isna().sum()/series.count()*100))
    return print(f'{nan_percentage}% of values are NaN')

def traffic_percent_by_state(df):
    '''
    Returns dataframe with total aircraft movements and percentage of totals
    Note that only works when all states are present - needs a sufficiently large sample
    '''
    df['dest_state'] = df.dest_city_name.str[-2:]
    df['origin_state'] = df.origin_city_name.str[-2:]
    movements = df[['origin_state', 'fl_date']].groupby('origin_state').count()['fl_date'] + df[['dest_state', 'fl_date']].groupby('dest_state').count()['fl_date']
    movements = pd.DataFrame(movements)
    movements['percentage_of_total'] = movements.fl_date/movements.fl_date.sum()*100
    return movements

def cumulative_percentage(df, threshold):
    '''
    Returns airports that cunmulatively consist of threshold percent of total traffic
    '''
    traffic = traffic_percent_by_state(df).sort_values(by='percentage_of_total',   ascending=False)
    return traffic[traffic.cumsum().percentage_of_total <= threshold].rename({'fl_date':'total_movements'}, axis=1)

def calculateSpeed(df):
    '''
    Returns column with speed values in MPH
    '''
    df['speed'] = df.distance/df.air_time*60
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
    data = otp(df)
    stat, p_value = stats.ttest_ind(data[data.OTP.isin([0])].speed,data[data.OTP.isin([1])].speed, nan_policy='omit')
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