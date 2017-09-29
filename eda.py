import pandas as pd
import numpy as np
import pyodbc
import sys


def enbase_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT GDD.assetWellFlac, GDD.gaugeDate, GDD.liquidGaugeFeet,
            GDD.liquidGaugeInches, GDD.liquidGaugeQuarter, GDD.waterGaugeFeet,
            GDD.waterGaugeInches, GDD.waterGaugeQuarter, PF.GUID
        FROM Enbase.GaugeDataDetailed AS GDD
        JOIN EnergySys.PR_FACILITY AS PF
        ON GDD.assetAssetName = PF.NAME
        ORDER BY GDD.assetWellFlac DESC, GDD.gaugeDate ASC;
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

def rtr_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT DDH.Well1_WellFlac, DDH.DateTime, DDH.Meter1_DiffPress,
               DDH.Meter1_DiffPressPDayAvg, DDH.Meter1_VolumeCDay,
               DDH.Meter1_VolumePDay, DDH.Meter1_StaticPress,
               DDH.Meter1_StaticPressPDayAvg, DDH.Meter1_FlowRate,
               DDH.Meter1_Temperature
        FROM RTR.DataDailyHistory AS DDH;
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

def rtr_clean(df, single=False):
    df_out = df[df['Well1_WellFlac'].notnull()]
    max_rows = 0
    max_well = None
    for well in df_out['Well1_WellFlac'].unique():
        row_count = df_out[df_out['Well1_WellFlac'] == well].shape[0]
        if row_count > max_rows:
            max_rows = row_count
            max_well = well
    if single == True:
        df_out = df_out[df_out['Well1_WellFlac'] == max_well]
    return df_out

def clean_data(df, single=False):
    df_out = df[(df['gaugeDate'].notnull())]
    max_rows = 0
    max_well = None
    # for well in df_out['assetWellFlac'].unique():
    #     row_count = df_out[df_out['assetWellFlac'] == well].shape[0]
    #     if row_count > max_rows:
    #         max_rows = row_count
    #         max_well = well
    # if single == True:
    #     df_out = df_out[df_out['assetWellFlac'] == max_well]
    df_out = df_out.drop_duplicates()
    return df_out

if __name__ == '__main__':
    df = rtr_fetch()
    df = rtr_clean(df, single=True)
