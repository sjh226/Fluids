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
        SELECT DW.API
        	  ,DW.WellName
              ,LGR.TotalOilOnSite AS LGROil
              ,LGR.TotalWaterOnSite AS LGRWater
              ,LGR.FacilityCapacity
              ,LGR.CalcDate
              ,LGR.PredictionMethod
        FROM [TeamOptimizationEngineering].[dbo].[InventoryAll] AS LGR
        JOIN (SELECT	FacilityKey
        				,MAX(CalcDate) maxtime
        		FROM [TeamOptimizationEngineering].[dbo].[InventoryAll]
        		GROUP BY FacilityKey, DAY(CalcDate), MONTH(CalcDate), YEAR(CalcDate)) AS MD
        	ON	MD.FacilityKey = LGR.FacilityKey
        	AND	MD.maxtime = LGR.CalcDate
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
        	ON LGR.FacilityKey = DW.FacilityKey
        JOIN [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
        	ON PTD.API = DW.API;
    """)

    # WHERE	LGR.FacilityKey IN (
	# 	SELECT FacilityKey
	# 	FROM [TeamOptimizationEngineering].[dbo].[InventoryAll]
	# 	GROUP BY FacilityKey
	# 	HAVING	SUM(TotalOilOnSite) > 0
	# 		AND	SUM(TotalWaterOnSite) > 0
	# 		AND CAST(MAX(CalcDate) AS DATE) = CAST(GETDATE() AS DATE)
	# 		AND COUNT(*) >= 31)

    cursor.execute(SQLCommand)
    results = cursor.fetchall()

    df = pd.DataFrame.from_records(results)
    connection.close()

    try:
    	df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    except:
    	df = None
    	print('Dataframe is empty')

    df['CalcDate'] = pd.DatetimeIndex(df['CalcDate']).normalize()

    return df.drop_duplicates()

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


if __name__ == '__main__':
    # df_lgr = lgr_pull()
    # df_gwr = gwr_pull()
    # df_lgr.to_csv('data/lgr.csv')
    # df_gwr.to_csv('data/gwr.csv')

    df_lgr = pd.read_csv('data/lgr.csv')
    # df_gwr = pd.read_csv('data/gwr.csv')

    # plot_lgr(df_lgr[(df_lgr['LGROil'] <= df_lgr['FacilityCapacity']) & \
    #                 (df_lgr['LGROil'] > 0)], variable='oil', plot_type='hist')
    # plot_lgr(df_lgr[(df_lgr['LGROil'] > df_lgr['FacilityCapacity']) & \
    #                 (df_lgr['LGROil'] < 1000)], variable='oil', plot_type='hist')

    lgr_over(df_lgr[df_lgr['LGROil'] > df_lgr['FacilityCapacity']])

    # df = data_link(df_lgr, df_gwr)
    # df.to_csv('data/linked_df.csv')
