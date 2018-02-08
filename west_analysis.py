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
        SELECT *
        FROM [TeamOptimizationEngineering].[dbo].[InventoryAll]
        WHERE BusinessUnit = 'West';
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

def shift_volumes(df):
	result_df = pd.DataFrame(columns=df.columns)
	for fac in df['FacilityKey'].unique():
		fac_df = df[df['FacilityKey'] == fac].sort_values('CalcDate')
		fac_df.loc[:, 'oil_rate'] = fac_df.loc[:, 'TotalOilOnSite'] - fac_df.loc[:, 'TotalOilOnSite'].shift(-1)
		result_df = result_df.append(fac_df)
	return result_df


if __name__ == '__main__':
    df = lgr_pull()
    df = shift_volumes(df)
