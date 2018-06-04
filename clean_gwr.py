import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import iqr, sem
import pyodbc
import sys
import sqlalchemy
import urllib
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time


def gwr_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=TeamOperationsAnalytics;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
        print("Connection Error")
        sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT  W.Facilitykey
                ,CONVERT(DATETIME, LEFT(REPLACE(GWR.DateTime,' ','T'), 19), 0) AS time
                ,SUM(CASE WHEN Tank LIKE '%CND%' THEN CAST(Value AS FLOAT)
                        WHEN Tank LIKE '%TOT%' THEN CAST(Value AS FLOAT) ELSE 0 END) * 20 AS 'CND'
                ,SUM(CASE WHEN Tank LIKE '%WAT%' THEN CAST(Value AS FLOAT) ELSE 0 END) * 20 AS 'WAT'
        FROM    (SELECT *
                FROM [TeamOperationsAnalytics].[dbo].[North_GWR]
                WHERE ISNUMERIC(Value) = 1) AS GWR
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] PTD
            ON PTD.TAG = GWR.Tag_Prefix
        JOIN [OperationsDataMart].[Dimensions].[Wells] W
            ON W.API = PTD.API
        WHERE CONVERT(DATETIME, LEFT(REPLACE(GWR.DateTime,' ','T'), 19), 0) >= DATEADD(day, DATEDIFF(day, 1, GETDATE()), 0)
            AND CONVERT(DATETIME, LEFT(REPLACE(GWR.DateTime,' ','T'), 19), 0) < DATEADD(day, DATEDIFF(day, 0, GETDATE()), 0)
        GROUP BY W.Facilitykey, CONVERT(DATETIME, LEFT(REPLACE(GWR.DateTime,' ','T'), 19), 0)
        ORDER BY W.Facilitykey, CONVERT(DATETIME, LEFT(REPLACE(GWR.DateTime,' ','T'), 19), 0);
	""")

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
        df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
        df.columns = [col.lower() for col in df.columns]
    except:
        df = None
        print('Dataframe is empty')

    return df

def turbine_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=TeamOperationsAnalytics;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
        print("Connection Error")
        sys.exit()

    cursor = connection.cursor()

    SQLCommand = ("""
        DROP TABLE IF EXISTS #Turbine;
        DROP TABLE IF EXISTS #Oil;
        DROP TABLE IF EXISTS #Water;
    """)

    cursor.execute(SQLCommand)

    SQLCommand = ("""
        SELECT  T.Tag_Prefix
                ,T.Tag
                ,T.DateTime
                ,T.Value AS Volume
                ,PTD.API
        		,W.Facilitykey
        INTO #Turbine
        FROM [TeamOperationsAnalytics].[dbo].[North_Turbine] T
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] PTD
          ON PTD.TAG = T.Tag_Prefix
        JOIN [OperationsDataMart].[Dimensions].[Wells] W
          ON W.API = PTD.API
        WHERE CONVERT(DATETIME, T.DateTime, 0) >= DATEADD(day, DATEDIFF(day, 1, GETDATE()), 0)
          AND CONVERT(DATETIME, T.DateTime, 0) < DATEADD(day, DATEDIFF(day, 0, GETDATE()), 0);
    """)

    cursor.execute(SQLCommand)

    SQLCommand = ("""
        SELECT	Facilitykey
        		,Tag
        		,SUM(Volume) AS FacOil
        INTO #Oil
        FROM #Turbine
        WHERE Tag LIKE '%CTS%'
        GROUP BY Facilitykey, Tag;

        SELECT	Facilitykey
        		,Tag
        		,SUM(Volume) AS FacWater
        INTO #Water
        FROM #Turbine
        WHERE Tag LIKE '%WAT%'
        GROUP BY Facilitykey, Tag;
    """)

    cursor.execute(SQLCommand)

    SQLCommand = ("""
        SELECT	T.Tag_Prefix
                ,T.Tag
                ,T.DateTime
                ,T.Volume
                ,T.API
        		,T.Facilitykey
        		,O.FacOil
        		,W.FacWater
        		,CASE	WHEN FacOil IS NOT NULL
        				THEN Volume / NULLIF(FacOil, 0)
        				WHEN FacWater IS NOT NULL
        				THEN Volume / NULLIF(FacWater, 0)
        		 ELSE NULL END AS Perc
        FROM #Turbine T
        LEFT OUTER JOIN #Oil O
          ON T.Facilitykey = O.Facilitykey
          AND T.Tag = O.Tag
        LEFT OUTER JOIN #Water W
          ON T.Facilitykey = W.Facilitykey
          AND T.Tag = W.Tag;
	""")

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
        df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
        df.columns = [col.lower() for col in df.columns]
    except:
        df = None
        print('Dataframe is empty')

    return df

def sql_push(df, table):
    params = urllib.parse.quote_plus('Driver={SQL Server Native Client 11.0};\
									 Server=SQLDW-L48.BP.Com;\
									 Database=TeamOperationsAnalytics;\
									 trusted_connection=yes'
                                     )
    engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

    df.to_sql(table, engine, schema='dbo', if_exists='append', index=False)

def outlier_regression(df, tank_type):
    lr = LinearRegression()
    poly = PolynomialFeatures(5)
    x_poly = poly.fit_transform(df['days'].values.reshape(-1, 1))
    lr = lr.fit(x_poly, df[tank_type])
    y = lr.predict(x_poly)
    dev = np.std(abs(df[tank_type] - y))
    if (dev != 0) & (df[(abs(df[tank_type] - y) <= 1.96 * dev)].shape[0] != 0):
        return df[(abs(df[tank_type] - y) <= 1.96 * dev) & \
                  (df[tank_type].notnull())] \
            [['tag_prefix', 'time', tank_type, 'tankcnt', 'days', 'volume']]
    else:
        return df[['tag_prefix', 'time', tank_type, 'tankcnt', 'days', 'volume']]

def conf_int(vals, confidence=.95):
    m, se = np.median(vals), sem(vals)
    h = se * sp.stats.t._ppf((1+confidence)/2., len(vals)-1)
    return m, h

def rebuild(df):
    return_df = pd.DataFrame(columns=['Facilitykey', 'DateKey', 'TANK_TYPE', \
                                      'TANKLVL', 'predict', 'rate2', \
                                      'CalcDate', 'Volume'])

    # Convert DateKey into days since first day
    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    day_min = df['time'].min()
    df.loc[:, 'days'] = (df['time'] - day_min).dt.total_seconds() / (24 * 60 * 60)

    # Loop through the same model building process for water, oil, and total
    for tank_type in ['oil', 'water']:
        if not df[df[tank_type].notnull()].empty:
            full_df = df.loc[df[tank_type].notnull(), :]

            # Block for removing outliers with linear regression
            # full_df = outlier_regression(full_df, tank_type)

            # Calculate an initial rate from nonnull values
            full_df.loc[:, 'rate'] = (full_df[tank_type] - \
                                      full_df[tank_type].shift(1)) / \
                                     ((full_df['time'] - \
                                       full_df['time'].shift(1)) / \
                                      np.timedelta64(1, 'm'))

            # Run nonnull, nonzero values through a confidence interval calculation
            vals = full_df.loc[(full_df['rate'].notnull()) & \
                               (full_df['rate'] != 0), 'rate'].values
            m, h = conf_int(vals)

            # Limit values to only include those with rates within the
            # calculated confidence interval
            if m > 0 and full_df[full_df['rate'] != 0].shape[0] > 10:
                value_limited_df = full_df.loc[(full_df['rate'] > (m-h)) & \
                                               (full_df['rate'] < (m+h)), :]
            else:
                value_limited_df = full_df.loc[:,:]

            # Remove any negative fluctuations
            rate_limited_df = value_limited_df.loc[value_limited_df['rate'] >= 0, :]

            # Calculate second rates off of these filtered values
            rate_limited_df.loc[:, 'rate2'] = \
                        (rate_limited_df[tank_type] - \
                         rate_limited_df[tank_type].shift(1)) / \
                        ((rate_limited_df['time'] - \
                          rate_limited_df['time'].shift(1)) / \
                         np.timedelta64(1, 'm'))

            # Calculate another confidence interval based on the filtered values
            # and limit the rates a final time
            vals = rate_limited_df.loc[(rate_limited_df['rate2'].notnull()) & \
                                       (rate_limited_df['rate2'] != 0), 'rate2'].values
            m, h = conf_int(vals)

            if m > 0 and rate_limited_df[rate_limited_df['rate2'] != 0].shape[0] > 10:
                rate_limited_df = rate_limited_df.loc[(rate_limited_df['rate2'] > (m-h)) & \
                                                      (rate_limited_df['rate2'] < (m+h))]

            # Fill in any 0 or empty rates with surrounding rates (forward before
            # backwards)
            rate_limited_df.loc[rate_limited_df['rate2'] <= 0, 'rate2'] = np.nan
            rate_limited_df['rate2'].fillna(method='ffill', inplace=True)
            rate_limited_df['rate2'].fillna(method='bfill', inplace=True)

            # Limit columns for both dataframes before merging and backfill nan
            full_df = full_df.loc[:, ['facilitykey', 'time', tank_type]]
            rate_limited_df = rate_limited_df.loc[:, ['facilitykey', 'time', tank_type, 'rate2']]
            type_df = pd.merge(full_df, rate_limited_df, \
                               how='left', on=['time', 'facilitykey', tank_type])
            type_df.fillna(method='ffill', inplace=True)
            type_df.fillna(method='bfill', inplace=True)

            # Fill in tank types depending on which iteration we're on
            if tank_type == 'oil':
                type_df.loc[:, 'TANK_TYPE'] = np.full(type_df.shape[0], 'CND')
            if tank_type == 'water':
                type_df.loc[:, 'TANK_TYPE'] = np.full(type_df.shape[0], 'WAT')

            # Fill in columns to match those expected in SQLDW and append this
            # to the return dataframe
            type_df.rename(index=str, columns={'facilitykey': 'Facilitykey', \
                                               'time': 'DateKey', \
                                               tank_type: 'TANKLVL', \
                                               'rate2': 'Rate'}, inplace=True)
            type_df.loc[:, 'CalcDate'] = type_df.loc[:, 'DateKey']
            return_df = return_df.append(type_df)

    return_df = return_df[['Facilitykey', 'DateKey', 'TANK_TYPE', 'TANKLVL', \
                           'Rate', 'CalcDate']]

    return return_df.sort_values(['Facilitykey', 'DateKey'])

def build_loop(df, tic_df=None):
    r_df = pd.DataFrame(columns=['Facilitykey', 'DateKey', 'TANK_TYPE', \
                                 'TANKLVL', 'TANKCNT', 'CalcDate', 'Volume'])

    # Loop through each unique tag and run data through cleaning
    for well in df['facilitykey'].unique():
        rwell_df = rebuild(df[df['facilitykey'] == well])
        r_df = r_df.append(rwell_df)
    return r_df

def clean_rate(sql=True):
    gwr_df = gwr_pull()
    gwr_df.rename(index=str, columns={'datetime':'time', 'wat':'water', \
                                     'cnd':'oil'}, inplace=True)

    clean_rate_df = build_loop(gwr_df)
    turb_df = turbine_pull()
    turb_df['datetime'] = pd.to_datetime(turb_df['datetime'])

    gwr_oil = clean_rate_df.loc[(clean_rate_df['TANK_TYPE'] == 'CND') & \
                                (clean_rate_df['DateKey'].dt.date == date.today() - timedelta(1)), \
                               ['DateKey', 'Facilitykey', 'TANK_TYPE', 'Rate']]
    gwr_wat = clean_rate_df.loc[(clean_rate_df['TANK_TYPE'] == 'WAT') & \
                                (clean_rate_df['DateKey'].dt.date == date.today() - timedelta(1)), \
                               ['DateKey', 'Facilitykey', 'TANK_TYPE', 'Rate']]

    turb_oil = turb_df.loc[(turb_df['tag'] == 'CTS_VY') & \
                           (turb_df['datetime'].dt.date == date.today() - timedelta(1)), \
                          ['api', 'facilitykey', 'datetime', 'perc']]
    turb_wat = turb_df.loc[(turb_df['tag'] == 'WAT_VY') & \
                           (turb_df['datetime'].dt.date == date.today() - timedelta(1)), \
                          ['api', 'facilitykey', 'datetime', 'perc']]

    oil_df = turbine_comp(gwr_oil, turb_oil)
    oil_df.rename(index=str, columns={'api': 'API', 'Facility': 'Facilitykey', \
                                      'Date': 'DateKey', 'Vol': 'Rate'}, inplace=True)
    oil_df['Tank_Type'] = 'CND'

    wat_df = turbine_comp(gwr_wat, turb_wat)
    wat_df.rename(index=str, columns={'api': 'API', 'Facility': 'Facilitykey', \
                                      'Date': 'DateKey', 'Vol': 'Rate'}, inplace=True)
    wat_df['Tank_Type'] = 'WAT'

    contr_df = oil_df.append(wat_df)
    contr_df.drop_duplicates(inplace=True)

    if sql:
        sql_push(contr_df, 'cleanGWR')

    return contr_df

def turbine_comp(gwr_df, turb_df):
    gwr_df['DateKey'] = pd.to_datetime(gwr_df['DateKey']).dt.date

    gwr_day_df = gwr_df.groupby(['DateKey', 'Facilitykey', 'TANK_TYPE'], \
                                as_index=False).mean()
    gwr_day_df.loc[:, 'Rate'] = gwr_day_df.loc[:, 'Rate'] * 60 * 24
    gwr_day_df['DateKey'] = pd.to_datetime(gwr_day_df['DateKey'])

    contr_df = turb_df.merge(gwr_day_df, how='outer', \
                             left_on=['facilitykey', 'datetime'], \
                             right_on=['Facilitykey', 'DateKey'])
    contr_df['VolRate'] = contr_df.loc[:, 'Rate'] * contr_df.loc[:, 'perc']

    contr_df['Facility'] = contr_df['Facilitykey'].fillna(contr_df['facilitykey'])
    contr_df['Date'] = contr_df['DateKey'].fillna(contr_df['datetime'])
    contr_df['Vol'] = contr_df['VolRate'].fillna(contr_df['Rate'])

    return contr_df[['api', 'Facility', 'Date', 'Vol']]

if __name__ == '__main__':
    clean_rate_df = clean_rate(sql=True)
