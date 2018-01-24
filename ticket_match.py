import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import sys
import cx_Oracle
from fluids_comp import gauge_pull


def tag_dict():
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
        SELECT  PTD.TAG AS tag_prefix
                ,PTD.API
                ,DF.Facilitykey
                ,DF.FacilityCapacity
        FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
        	ON PTD.API = DW.API
        JOIN [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] AS DF
        	ON DW.Facilitykey = DF.Facilitykey
        GROUP BY PTD.TAG, PTD.API, DF.Facilitykey, DF.FacilityCapacity;
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

    return df.drop_duplicates()

def tank_count():
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
        SELECT DT.Facilitykey
        	   ,COUNT(DT.Tankkey) AS tank_count
        FROM [TeamOptimizationEngineering].[dbo].[DimensionsTanks] DT
        WHERE DT.BusinessUnit = 'North'
        GROUP BY DT.Facilitykey;
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

    return df.drop_duplicates()

def ticket_pull():
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
        SELECT CAST(RT.runTicketStartDate AS DATE) AS date
              ,RT.ticketType
              ,RT.tankCode
        	  ,DT.Facilitykey
              ,RT.grossVolume
          FROM [EDW].[Enbase].[RunTicket] RT
          JOIN [TeamOptimizationEngineering].[dbo].[DimensionsTanks] DT
        	ON RT.tankCode = DT.TankCode
         WHERE DT.BusinessUnit = 'North';
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

    df['date'] = pd.to_datetime(df['date'])

    return df.drop_duplicates()

def map_tag(vol, tag):
    df = vol.merge(tag, on='tag_prefix', how='inner')
    # df = df.drop(['tag_prefix', 'API'], axis=1)
    df = df.dropna()
    df['oil_rate'] = df['oil'] - df['oil'].shift(1)
    df.loc[df['oil_rate'] < 0, 'oil_rate'] = np.nan
    df['oil_rate']
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby(['Facilitykey', 'time', 'FacilityCapacity', 'tag_prefix', 'tankcnt'], as_index=False).mean()
    df = df.groupby(['Facilitykey', 'time', 'FacilityCapacity', 'tag_prefix'], as_index=False).max()
    return df.sort_values(['Facilitykey', 'time'])

def total_plot(df, t_df):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    facility = df['Facilitykey'].unique()[0]
    capacity = df['FacilityCapacity'].unique()[0]
    t_df = t_df[t_df['date'] >= df['time'].min()]
    water = t_df[t_df['ticketType'] == 'Water Haul']
    oil = t_df[t_df['ticketType'] == 'Oil Haul']

    ax.plot(df['time'], df['oil'], label='GWR Volume')
    # i = 0
    # for date in water['date']:
    #     ax.axvline(date, color='blue', linestyle='--', label='Water Haul' if i == 0 else '')
    #     i += 1
    i = 0
    for date in oil['date']:
        ax.axvline(date, color='red', linestyle='--', label='Oil Haul' if i == 0 else '')
        i += 1

    plt.title('Oil GWR Volumes for Facility {}'.format(facility))
    plt.xlabel('Date')
    plt.ylabel('bbl')

    ymin, ymax = plt.ylim()
    if ymin > 0:
        plt.ylim(ymin=0)

    plt.xticks(rotation='vertical')
    plt.legend()
    plt.tight_layout()

    plt.savefig('images/gwr/oil/oil_{}.png'.format(facility))

def plot_rate(rate_df):
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    df = rate_df[['time', 'oil_rate']]
    df.loc[df['oil_rate'] < 0, 'oil_rate'] = np.nan
    df.dropna(inplace=True)

    facility = rate_df['Facilitykey'].unique()[0]

    ax.plot(df['time'], df['oil_rate'])

    plt.title('Oil Rates for Facility {}'.format(facility))
    plt.xlabel('Date')
    plt.ylabel('bbl/day')

    plt.xticks(rotation='vertical')
    plt.savefig('images/rates/total/tot_rate_{}.png'.format(facility))

def tank_merge(pi_df, sql_df):
    pi_df['gwr_tanks'] = pi_df['tankcnt']
    pi_df = pi_df[['Facilitykey', 'tankcnt']]
    df = pi_df.merge(sql_df, on='Facilitykey', how='outer')
    df.fillna(0, inplace=True)
    return df.drop_duplicates()

def get_rate(df):
    result_df = pd.DataFrame(columns=df.columns)
    for tank in df['tag_prefix'].unique():
        tank_df = df[df['tag_prefix'] == tank].sort_values('time')
        tank_df['oil_rate'] = tank_df['oil'] - tank_df['oil'].shift(1)
        result_df = result_df.append(tank_df)
    return result_df


if __name__ == '__main__':
    tag_df = tag_dict()
    vol_df = pd.read_csv('data/nan_vol_df.csv')
    df = map_tag(vol_df, tag_df)

    tank_df = tank_count()
    tank_df = tank_merge(df, tank_df)
    match_df = tank_df[tank_df['tankcnt'] == tank_df['tank_count']]

    gauges = gauge_pull()
    ticket_df = ticket_pull()

    # rate_df = get_rate(df)
    # rate_df.to_csv('data/gwr_oil_rate.csv')
    rate_df = pd.read_csv('data/gwr_oil_rate.csv')

    for facility in match_df['Facilitykey'].unique():
        total_plot(df[df['Facilitykey'] == facility], ticket_df[ticket_df['Facilitykey'] == facility])
    #     break

    # for facility in match_df['Facilitykey'].unique():
    #     plot_rate(df[df['Facilitykey'] == facility])
        # break
