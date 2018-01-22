import cx_Oracle
import numpy as np
import pandas as pd


def data_conn():
	connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

	cursor = connection.cursor()
	query = ("""
		SELECT  TAG_PREFIX
				,TRUNC(TIME) AS my_date
				,MAX(CTS_VC)
		FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
		WHERE CTS_VC IS NOT NULL
		GROUP BY TAG_PREFIX, TRUNC(TIME)
		ORDER BY TAG_PREFIX, TRUNC(TIME)
	""")

	cursor.execute(query)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
		df.columns = [col.lower() for col in df.columns]
	except:
		df = None
		print('Dataframe is empty')

	cursor.close()
	connection.close()

	return df

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


if __name__ == "__main__":
	df = data_conn()
	df.to_csv('data/turbine.csv')

	df = pd.read_csv('data/turbine.csv')

	tag_df = tag_dict()
