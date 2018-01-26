import cx_Oracle
import pyodbc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ticket_match import turbine_gwr_pull


def data_conn():
	connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

	cursor = connection.cursor()
	query = ("""
		SELECT  TAG_PREFIX
				,TRUNC(TIME) AS flow_date
				,MAX(CTS_VC) AS volume
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

def shift_volumes(df):
	result_df = pd.DataFrame(columns=df.columns)
	for tag in df['tag_prefix']:
		tag_df = df[df['tag_prefix'] == tag]
		tag_df.loc[:, 'volume'] = tag_df.loc[:, 'volume'].shift(-1)
		result_df = result_df.append(tag_df)
	return result_df

def map_tag(vol, tag):
	df = vol.merge(tag, on='tag_prefix', how='inner')
	df = df.drop(['Unnamed: 0', 'tag_prefix', 'API'], axis=1)
	df = df.dropna()
	# df['oil_rate'] = df['volume'] - df['oil'].shift(1)
	df['my_date'] = pd.to_datetime(df['my_date'])
	df = df.groupby(['Facilitykey', 'my_date', 'FacilityCapacity'], as_index=False).sum()
	return df.sort_values(['Facilitykey', 'my_date'])

def plot_rate(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	facility = df['Facilitykey'].unique()[0]

	ax.plot(df['my_date'], df['volume'])

	plt.title('Liquid Rates for Facility {}'.format(facility))
	plt.xlabel('Date')
	plt.ylabel('bbl/day')
	plt.xticks(rotation='vertical')

	plt.savefig('images/turbine/rate_{}.png'.format(facility))


if __name__ == "__main__":
	df = data_conn()
	df = shift_volumes(df)
	# df.to_csv('data/turbine.csv')

	gwr_df = turbine_gwr_pull()
	# vol_df = pd.read_csv('data/turbine.csv')
	# tag_df = tag_dict()
	# df = map_tag(vol_df, tag_df)

	# for facility in df['Facilitykey'].unique():
	# 	plot_rate(df[df['Facilitykey'] == facility])
		# break
