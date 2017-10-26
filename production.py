import pyodbc
import pandas as pd
import numpy as np


def prod_query():
    try:
        connection = pyodbc.connect(Driver = '{SQL Server Native Client 11.0};Server=SQLDW-TEST-L48.BP.Com;Database=OperationsDataMart;trusted_connection=yes')
    except pyodbc.Error:
    	print("Connection Error")
    	sys.exit()

    cursor = connection.cursor()
    SQLCommand = ("""
        SELECT P.Wellkey
              ,W.FacilityKey
              ,W.WellFlac
              ,P.Oil
              ,P.Gas
              ,P.Water
              ,P.DateKey
        FROM [OperationsDataMart].[Facts].[Production] AS P
        JOIN [OperationsDataMart].[Dimensions].[Wells] AS W
            ON P.Wellkey = W.Wellkey;
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


if __name__ == '__main__':
    prod_query()
