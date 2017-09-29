import pandas as pd
import numpy as np
import pyodbc
import sys


def data_fetch():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=EDW;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT Top 1000 GDD.assetWellFlac AS WellFlac, GDD.gaugeDate, GDD.liquidGaugeFeet,
            GDD.liquidGaugeInches, GDD.liquidGaugeQuarter, GDD.waterGaugeFeet,
            GDD.waterGaugeInches, GDD.liquidGaugeQuarter, GDD.TankHeight,
            DDH.DateTime
        FROM Enbase.GaugeDataDetailed AS GDD
        JOIN RTR.DataDailyHistory AS DDH
        ON GDD.assetWellFlac = DDH.Well1_WellFlac
        ORDER BY GDD.assetWellFlac DESC, DDH.DateTime ASC;
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

if __name__ == '__main__':
    df = data_fetch()
