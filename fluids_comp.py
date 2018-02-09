import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import sys


def lgr_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=TeamOptimizationEngineering;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT  LGR.FacilityKey
                ,LGR.FacilityName
                ,AVG(LGR.TotalOilOnSite) AS LGROil
                ,AVG(LGR.TotalWaterOnSite) AS LGRWater
                ,LGR.FacilityCapacity
                ,LGR.CalcDate
                ,LGR.PredictionMethod
        FROM [TeamOptimizationEngineering].[dbo].[InventoryAll_Calculated] AS LGR
        JOIN (SELECT	FacilityKey
        				,MAX(CalcDate) maxtime
        		FROM [TeamOptimizationEngineering].[dbo].[InventoryAll_Calculated]
        		GROUP BY FacilityKey, CAST(CalcDate AS DATE)) AS MD
        	ON	MD.FacilityKey = LGR.FacilityKey
        	AND	MD.maxtime = LGR.CalcDate
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
        	ON LGR.FacilityKey = DW.FacilityKey
        WHERE LGR.BusinessUnit = 'North'
            --AND LGR.PredictionMethod = 'LGRv4'
        GROUP BY LGR.FacilityKey, LGR.FacilityName, LGR.FacilityCapacity, LGR.CalcDate, LGR.PredictionMethod;
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    df['CalcDate'] = pd.to_datetime(pd.DatetimeIndex(df['CalcDate']).normalize())

    return df.drop_duplicates()

def spill_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=OperationsDataMart;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT FI.Facilitykey AS FacilityKey
              ,FI.FacilityName
              ,FI.DateKey AS Date
              ,FI.LastGaugeDate
              ,FI.DaysSinceLastGauge
              ,FI.Oil
              ,FI.Water
          FROM [OperationsDataMart].[Reporting].[FacilityInventory] AS FI
          WHERE FI.Asset = 'West'
          GROUP BY FI.Facilitykey, FI.FacilityName, FI.DateKey,
                   FI.LastGaugeDate, FI.DaysSinceLastGauge, FI.Oil, FI.Water
          ORDER BY FI.Facilitykey, FI.DateKey
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    df['Date'] = pd.to_datetime(df['Date'])

    return df.drop_duplicates()

def bad_gauge_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=OperationsDataMart;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT  TD.TankID
                ,T.Facilitykey
                ,TD.BusinessUnit
                ,TD.DateKey
                ,TD.DateTime
                ,TD.RecordType
                ,TD.CloseOil
                ,TD.CloseWater
                ,TD.CreatedDate
        FROM [OperationsDataMart].[Stage].[TankDispositions] AS TD
        JOIN [OperationsDataMart].[Dimensions].[Tanks] AS T
	       ON T.TankID =  TD.TankID
        WHERE TD.BusinessUnit = 'North';
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    return df

def gwr_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=TeamOptimizationEngineering;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT	PIT.TAG_PREFIX
                ,API
        		,TANK_TYPE
        		,TANKLVL
        		,CalcDate
        FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks] AS PIT
        JOIN (SELECT	TAG_PREFIX
        				,MAX(CalcDate) maxtime
        	FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        	GROUP BY TAG_PREFIX, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        ON	MD.TAG_PREFIX = PIT.TAG_PREFIX
        AND	MD.maxtime = PIT.CalcDate
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
	        ON PTD.TAG = PIT.TAG_PREFIX;
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    new_df = df[['TAG_PREFIX', 'API', 'CalcDate', 'TANKLVL']]
    new_df = new_df.groupby(['TAG_PREFIX', 'API', 'CalcDate'], as_index=False).sum()

    def water(row):
        wat = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'WAT')]['TANKLVL'].values
        try:
            val = wat[0]
        except:
            val = np.nan
        return val

    def cond(row):
        cond = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'CND')]['TANKLVL'].values
        try:
            val = cond[0]
        except:
            val = np.nan
        return val

    def total(row):
        tot = df[(df['TAG_PREFIX'] == row['TAG_PREFIX']) & (df['CalcDate'] == row['CalcDate'])\
                 & (df['TANK_TYPE'] == 'TOT')]['TANKLVL'].values
        try:
            val = tot[0]
        except:
            val = np.nan
        return val

    new_df['water'] = new_df.apply(water, axis=1)
    new_df['oil'] = new_df.apply(cond, axis=1)
    new_df['total'] = new_df.apply(total, axis=1)
    new_df.loc[new_df['oil'].isnull(), 'oil'] = new_df[new_df['oil'].isnull()]['total'] - \
                                                new_df[new_df['oil'].isnull()]['water']

    new_df['CalcDate'] = pd.DatetimeIndex(new_df['CalcDate']).normalize()

    return new_df.drop_duplicates()

def ticket_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=EDW;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT	RT.assetId
                ,
        FROM [EDW].[Enbase].[RunTicket] AS RT
        JOIN (SELECT	TAG_PREFIX
        				,MAX(CalcDate) maxtime
        	FROM [TeamOptimizationEngineering].[Reporting].[PI_Tanks]
        	GROUP BY TAG_PREFIX, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        ON	MD.TAG_PREFIX = PIT.TAG_PREFIX
        AND	MD.maxtime = PIT.CalcDate
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
	        ON PTD.TAG = PIT.TAG_PREFIX;
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

def gauge_pull():
    try:
        connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
                                    r'Server=SQLDW-L48.BP.Com;'
                                    r'Database=EDW;'
                                    r'trusted_connection=yes'
                                    )
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SET NOCOUNT ON;
        DROP TABLE IF EXISTS #Tanks

        SELECT	F.Facilitykey
                ,F.TankCount
        		,GD.tankCode
        		,GD.gaugeDate
                ,F.FacilityCapacity
        		,((GD.liquidGaugeFeet + (GD.liquidGaugeInches / 12) + (GD.liquidGaugeQuarter / 48))
                 - (GD.waterGaugeFeet + (GD.waterGaugeInches / 12) + (GD.waterGaugeQuarter / 48))) * 20 AS oil
        		,(GD.waterGaugeFeet + (GD.waterGaugeInches / 12) + (GD.waterGaugeQuarter / 48)) * 20 AS water
        INTO #Tanks
        FROM EDW.Enbase.GaugeData AS GD
        JOIN OperationsDataMart.Dimensions.Tanks AS T
        	ON T.TankCode = GD.tankCode
        JOIN OperationsDataMart.Dimensions.Facilities AS F
            ON F.Facilitykey = T.Facilitykey
        WHERE F.BusinessUnit = 'North'
        ORDER BY T.Facilitykey, GD.gaugeDate;

        SELECT	Facilitykey
                ,TankCount
        		,CAST(gaugeDate AS DATE) AS gaugeDate
                ,FacilityCapacity
        		,SUM(oil) AS total_oil
        		,SUM(water) AS total_water
                ,COUNT(*) AS tanks
        FROM #Tanks
        WHERE oil IS NOT NULL
        GROUP BY Facilitykey, TankCount, FacilityCapacity, CAST(gaugeDate AS DATE)
        HAVING COUNT(*) = TankCount
        ORDER BY Facilitykey, CAST(gaugeDate AS DATE);
    """)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    df['gaugeDate'] = pd.DatetimeIndex(df['gaugeDate']).normalize()

    return df

def data_link(lgr, gwr):
    return lgr.merge(gwr, how='outer', on=['API', 'CalcDate'])

def plot_lgr(df, variable='oil', plot_type='hist'):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    var = 'LGR' + variable.title()

    ob_date = {}
    for date in df['CalcDate'].unique():
        ob_date[date] = df[df['CalcDate'] == date][var].mean()

    if plot_type == 'hist':
        plt.hist(df[var], bins=100, density=True)
        plt.title('LGR Within Facility Capacity')
        plt.xlabel('LGR Oil Value')
        plt.ylabel('Percent of Total')
        variable = 'oilinrange'
    elif plot_type == 'box':
        ax.boxplot([df['LGROil'], df['LGROil']])
        labels = ['Oil', 'Water']
        ax.set_xticklabels(labels)

    plt.savefig('images/lgr_{}_{}.png'.format(plot_type, variable))

def lgr_gauge_plot(lgr_df, gauge_df):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    date_min = lgr_df['CalcDate'].min()
    date_max = lgr_df['CalcDate'].max()
    g_df = gauge_df[(gauge_df['gaugeDate'] >= date_min) & (gauge_df['gaugeDate'] <= date_max)]
    # g_df = gauge_df

    try:
        capacity = gauge_df['FacilityCapacity'].unique()[0]
    except:
        capacity = 0
    version = lgr_df['PredictionMethod'].unique()[0]

    ax.plot(lgr_df['CalcDate'], lgr_df['LGROil'], label='LGR Value')
    ax.plot(g_df['gaugeDate'], g_df['total_oil'], 'ro', label='Real Gauges')
    ax.axhline(capacity, date_min, date_max, linestyle='--', color='#920f25', label='Facility Capacity')
    ymin, ymax = plt.ylim()

    facility = lgr_df['FacilityName'].unique()[0]
    plt.title('LGR for Facility {}'.format(facility))
    ax.set_xlabel('Date')
    ax.set_ylabel('bbl Oil')
    plt.ylim(ymin=0)
    plt.ylim(ymax=ymax + (ymax * .3))

    ymin, ymax = plt.ylim()
    ax.text(date_min, ymax - (ymax * .1), version, fontsize=18)

    cnt = 0
    if len(ax.xaxis.get_ticklabels()) > 12:
        for label in ax.xaxis.get_ticklabels():
            if cnt % 3 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
            cnt += 1

    plt.xticks(rotation='vertical')
    plt.legend()

    plt.savefig('images/lgr/worst/lgr_gauge_{}.png'.format(facility))

def lgr_over(df):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    v4_dic = {}
    v2_dic = {}
    for date in df['CalcDate'].unique():
        v2_dic[date] = df[(df['CalcDate'] == date) & (df['PredictionMethod'] == 'LGRv2')].shape[0]
        v4_dic[date] = df[(df['CalcDate'] == date) & (df['PredictionMethod'] == 'LGRv4')].shape[0]

    width = 0.8
    v2 = ax.bar(list(v2_dic.keys()), list(v2_dic.values()), width, color='#37b782', tick_label='Version 2')
    v4 = ax.bar(list(v4_dic.keys()), list(v4_dic.values()), width, color='#1d7f56', tick_label='Version 4')

    plt.title('LGR Predictions Over Facility Capacity by Version')
    ax.set_xticklabels(tuple(v2_dic.keys()), rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Count of Wells')
    plt.legend((v2[0], v4[0]), ('Version 2', 'Version 4'))
    plt.savefig('images/lgr_over_version.png')

def match_gauge(lgr, gauge):
    lgr['Date'] = pd.to_datetime(lgr['CalcDate']) + pd.Timedelta('1 days')
    lgr = lgr[['FacilityKey', 'FacilityName', 'Date', 'LGROil', 'LGRWater', 'PredictionMethod']]
    gauge['Date'] = gauge['gaugeDate']
    gauge['FacilityKey'] = gauge['Facilitykey']
    gauge = gauge[['FacilityKey', 'Date', 'total_oil', 'total_water']]
    df = lgr.merge(gauge, on=['FacilityKey', 'Date'], how='left')
    df['total_oil'] = df['total_oil'].astype(float)
    df['off_oil'] = abs(df['LGROil'] - df['total_oil'])
    df['per_off'] = abs(df['LGROil'] / df['total_oil'])
    df['per_err'] = (abs(df['total_oil'] - df['LGROil'])/df['total_oil']) * 100
    # df['per_off'] = np.where(df['per_off'] <= 1, df['per_off'], 1 - (df['per_off'] - 1))
    return df

def spill_gauge(spill, gauge):
    spill = spill[['FacilityKey', 'FacilityName', 'Date', 'Oil', 'Water']]
    gauge['Date'] = gauge['gaugeDate']
    gauge['FacilityKey'] = gauge['Facilitykey']
    gauge = gauge[['FacilityKey', 'Date', 'total_oil', 'total_water']]
    df = spill.merge(gauge, on=['FacilityKey', 'Date'], how='left')
    df['total_oil'] = df['total_oil'].astype(float)
    df['off_oil'] = abs(df['Oil'] - df['total_oil'])
    df['per_off'] = abs(df['Oil'] / df['total_oil'])
    df['per_err'] = (abs(df['total_oil'] - df['Oil'])/df['total_oil']) * 100
    return_df = pd.DataFrame(columns=['FacilityKey', 'FacilityName', \
                                      'delta_oil', 'average_delta', \
                                      'perc_diff', 'per_err'])
    for fac in df['FacilityKey'].unique():
        fac_df = df[df['FacilityKey'] == fac]
        return_df = return_df.append({'FacilityKey':fac, \
                                      'FacilityName':fac_df['FacilityName'].unique()[0], \
                                      'delta_oil':fac_df['off_oil'].sum(), \
                                      'average_delta':fac_df['off_oil'].mean(), \
                                      'perc_diff':fac_df['per_off'].mean(), \
                                      'per_err':fac_df['per_err'].mean()}, \
                                      ignore_index=True)
    return return_df.sort_values('average_delta')

def facility_error(df):
    return_df = pd.DataFrame(columns=['FacilityKey', 'FacilityName', \
                                      'delta_oil', 'average_delta', \
                                      'perc_diff', 'per_err', 'PredictionMethod'])
    for fac in df['FacilityKey'].unique():
        pred = max(df[df['FacilityKey'] == fac]['PredictionMethod'].unique())
        fac_df = df[(df['FacilityKey'] == fac) & (df['PredictionMethod'] == pred)]
        return_df = return_df.append({'FacilityKey':fac, \
                                      'FacilityName':fac_df['FacilityName'].unique()[0], \
                                      'delta_oil':fac_df['off_oil'].sum(), \
                                      'average_delta':fac_df['off_oil'].mean(), \
                                      'perc_diff':fac_df['per_off'].mean(), \
                                      'per_err':fac_df['per_err'].mean(), \
                                      'PredictionMethod':pred}, \
                                      ignore_index=True)
    return return_df.sort_values('average_delta')

def acc_distr(df, month=''):
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(df[(df['per_err'] != np.inf) & \
               (df['per_err'].notnull()) & \
               (df['per_err'] <= 200)]['per_err'].values, \
               bins=80, color='#2d92e5', label='LGR Error Rates')
    ax.axvline(20, color='#ad0f0f', linestyle='dashed', label='20% Error')
    error = df[(df['per_err'] != np.inf) & \
               (df['per_err'].notnull()) & \
               (df['per_err'] <= 100)]['per_err'].mean()
    ax.axvline(error, color='#540bc1', linestyle='dashed', label='Mean Error')

    plt.title('Percent Error of LGR')
    plt.xlabel('Percent Error (%)')
    plt.ylabel('Facility Count')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.text(xmax*.5, ymax*.8, 'Averaging {:.2f}% Error'.format(error), fontsize=18)
    plt.legend()

    plt.savefig('images/lgr/lgr_error_{}.png'.format(month))


if __name__ == '__main__':
    # df_lgr = lgr_pull()
    # df_lgr.to_csv('data/lgr.csv')
    df_lgr = pd.read_csv('data/lgr.csv')

    # df_spill = spill_pull()

    # df_gwr = gwr_pull()
    gauge_df = gauge_pull()

    # off_df = spill_gauge(df_spill, gauge_df)

    df = match_gauge(df_lgr, gauge_df[gauge_df['total_oil'] >= 50])
    # off_df = facility_error(df)
    # off_df.to_csv('data/temp_lgr_error.csv')
    # off_df = pd.read_csv('data/temp_lgr_error.csv')
    # worst_lgr = off_df[off_df['per_err'].notnull()].sort_values('per_err')
    # worst_lgr = worst_lgr.tail(20)
    # df_lgr = df_lgr[df_lgr['FacilityKey'].isin(worst_lgr['FacilityKey'].unique())]
    # df_lgr['CalcDate'] = pd.to_datetime(df_lgr['CalcDate'])

    # for month in [8, 9, 10, 11, 12, 1]:
    #     off_df = facility_error(df[df['Date'].dt.month == month])
    #     acc_distr(off_df[(off_df['PredictionMethod'] == 'LGRv4')], month=month)

    # df_lgr.to_csv('data/lgr.csv')
    # df_gwr.to_csv('data/gwr.csv')
    # gauge_df.to_csv('data/gauges.csv')

    # df_lgr = pd.read_csv('data/lgr.csv')
    # df_gwr = pd.read_csv('data/gwr.csv')
    # gauge_df = pd.read_csv('data/gauges.csv')

    # for facility in sorted(df_lgr['FacilityKey'].unique()):
    #     lgr_gauge_plot(df_lgr[df_lgr['FacilityKey'] == facility].sort_values('CalcDate'), \
    #                    gauge_df[(gauge_df['Facilitykey'] == facility) & \
    #                             (gauge_df['total_oil'] >= 140)].sort_values('gaugeDate'))
        # break

    # plot_lgr(df_lgr[(df_lgr['LGROil'] <= df_lgr['FacilityCapacity']) & \
    #                 (df_lgr['LGROil'] > 0)], variable='oil', plot_type='hist',\
    #                 gauge=gauge_df)
    # plot_lgr(df_lgr[(df_lgr['LGROil'] > df_lgr['FacilityCapacity']) & \
    #                 (df_lgr['LGROil'] < 1000)], variable='oil', plot_type='hist')

    # lgr_over(df_lgr[df_lgr['LGROil'] > df_lgr['FacilityCapacity']])

    # df = data_link(df_lgr, df_gwr)
    # df.to_csv('data/linked_df.csv')
