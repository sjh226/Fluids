import cx_Oracle
import pyodbc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gwr import turbine_gwr_pull


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
				,DF.FacilityName
		FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
			ON PTD.API = DW.API
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] AS DF
			ON DW.Facilitykey = DF.Facilitykey
		GROUP BY PTD.TAG, PTD.API, DF.Facilitykey, DF.FacilityName, DF.FacilityCapacity;
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
	for tag in df['tag_prefix'].unique():
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

def turb_contr(gwr_df, turbine_df):
	tag_list = ['WAM-ML11_150H', 'WAM-ML11_160H', 'WAM-ML11_160D', 'WAM-BB19', \
				'WAM-CL29_150H', 'WAM-CH320C1', 'WAM-HP13_150H', \
				'WAM-CL29_160H', 'WAM-LM8_115H']
	g_df = gwr_df[gwr_df['tag_prefix'].str.contains('|'.join(tag_list))]
	t_df = turbine_df[turbine_df['tag_prefix'].str.contains('|'.join(tag_list))]

	g_df = g_df[['Facilitykey', 'time', 'FacilityName', 'water', 'oil']]
	g_df['time'] = pd.DatetimeIndex(g_df['time']).normalize()
	g_df = g_df.groupby(['Facilitykey', 'FacilityName', 'time'], as_index=False).median()

	t_df.loc[:, 'time'] = pd.to_datetime(t_df['flow_date'])
	t_df = t_df[['tag_prefix', 'FacilityName', 'time', 'volume', 'API']]
	t_df.loc[:, 'contr'] = np.nan
	t_df.loc[:, 'oil'] = np.nan
	t_df.loc[:, 'water'] = np.nan
	t_df = t_df[t_df['time'] >= g_df['time'].min()]

	for fac in t_df['FacilityName'].unique():
		for time in t_df[t_df['FacilityName'] == fac]['time']:
			contr = t_df[(t_df['FacilityName'] == fac) & (t_df['time'] == time)]['volume'].sum()
			t_df.loc[(t_df['FacilityName'] == fac) & (t_df['time'] == time), 'contr'] = contr
			match_df = g_df[(g_df['FacilityName'] == fac) & (g_df['time'] == time)]
			if len(match_df['oil'].values) > 0:
				t_df.loc[(t_df['FacilityName'] == fac) & (t_df['time'] == time), 'oil'] = match_df['oil'].values[0]
			if len(match_df['water'].values) > 0:
				t_df.loc[(t_df['FacilityName'] == fac) & (t_df['time'] == time), 'water'] = match_df['water'].values[0]

	t_df.loc[:, 'oil_diff'] = t_df['oil'] - t_df['oil'].shift(1)
	t_df.loc[:, 'oil_contr'] = t_df['oil_diff'] * (t_df['volume'] / t_df['contr'])
	return t_df

def plot_contr(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	well = df['tag_prefix'].unique()[0].lstrip('WAM-')
	pos_df = df
	pos_df.loc[(pos_df['oil_contr'] < 0) | (pos_df['oil_contr'].isnull()), 'oil_contr'] = 0
	pos_df = pos_df[pos_df['oil_contr'] > 0]

	ax.plot(df['time'], df['volume'], color='black', label='Turbine Volume')
	ax.plot(pos_df['time'], pos_df['oil_contr'], color='red', label='GWR with Turbine Contribution')

	plt.xticks(rotation='vertical')
	plt.xlabel('Date')
	plt.ylabel('Oil Rate (bbl/day)')
	plt.title('Turbine Contribution for {}'.format(well))
	plt.legend()

	plt.savefig('images/contributions/cont_{}.png'.format(well))


if __name__ == "__main__":
	# df = data_conn()
	# df = shift_volumes(df)
	# df.to_csv('data/turbine.csv')
	# df = pd.read_csv('data/turbine.csv')

	# gwr_df = turbine_gwr_pull()
	# gwr_df.to_csv('data/turbine_gwr.csv')
	# gwr_df = pd.read_csv('data/turbine_gwr.csv')

	# tag_df = tag_dict()
	# turbine_df = df.merge(tag_df, on='tag_prefix', how='inner')
	# g_df = turb_contr(gwr_df, turbine_df)
	# g_df.to_csv('data/turbine_contribution.csv')

	g_df = pd.read_csv('data/turbine_contribution.csv')
	g_df['time'] = pd.to_datetime(g_df['time'])
	for well in g_df['tag_prefix'].unique():
		plot_contr(g_df[g_df['tag_prefix'] == well].sort_values('time'))

	# this = turbine_df[['API', 'Facilitykey']]
	# that = this.groupby('Facilitykey')['API'].nunique()
	# for fac in that[that > 5].index:
	# 	if fac in gwr_df['Facilitykey'].values and fac in turbine_df['Facilitykey'].values:
	# 		print('YES! at: ', fac)

	# vol_df = pd.read_csv('data/turbine.csv')
	# tag_df = tag_dict()
	# df = map_tag(vol_df, tag_df)

	# for facility in df['Facilitykey'].unique():
	# 	plot_rate(df[df['Facilitykey'] == facility])
		# break
