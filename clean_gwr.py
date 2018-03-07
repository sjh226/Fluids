import pandas as pd
import numpy as np
from scipy.stats import iqr, sem
import pyodbc
import sys
import cx_Oracle
import sqlalchemy
import urllib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time


def oracle_pull():
	connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

	cursor = connection.cursor()
	query = '''
		SELECT
			TAG_PREFIX,
			TIME,
			Tank_Type,
			TankVol,
			TankCnt

		FROM (
				 SELECT
					 row_number()
					 OVER (
						 PARTITION BY TAG_PREFIX
						 ORDER BY Time DESC ) AS rk,
					 TAG_PREFIX,
					 TIME,
					 (nvl(TNK_1_TOT_LVL, 0) + nvl(TNK_2_TOT_LVL, 0) +
					  nvl(TNK_3_TOT_LVL, 0) + nvl(TNK_4_TOT_LVL, 0) +
					  nvl(TNK_5_TOT_LVL, 0) + nvl(TNK_6_TOT_LVL, 0) +
					  nvl(TNK_7_TOT_LVL, 0) + nvl(TNK_8_TOT_LVL, 0) +
					  nvl(TNK_9_TOT_LVL, 0) + nvl(TNK_10_TOT_LVL, 0)) * 20 AS TankVol,
					 GAS_VC,
					 CASE WHEN (TNK_1_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_2_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_3_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_4_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_5_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_6_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_7_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_8_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_9_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_10_TOT_LVL) >= 0
						 THEN 1
					 ELSE 0 END               AS TankCnt,
					 'TOT'                    AS Tank_Type
				 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
			 Where TIME >= ADD_MONTHS(TRUNC(SYSDATE), -12)
			 ) Vol
		WHERE (Vol.TankVol > 0 AND TIME >= ADD_MONTHS(TRUNC(SYSDATE), -12))
		  AND (TAG_PREFIX LIKE 'WAM-CH320C1-160H%'
				OR TAG_PREFIX LIKE 'WAM-CH452K29150H%'
				  OR TAG_PREFIX LIKE 'WAM-CH533B3_80D%'
				OR TAG_PREFIX LIKE 'WAM-CL29_150H%'
				OR TAG_PREFIX LIKE 'WAM-CL29_160H%'
				OR TAG_PREFIX LIKE 'WAM-HP13_150%'
				OR TAG_PREFIX LIKE 'WAM-LM8_115H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_150H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_160H%'
				OR TAG_PREFIX LIKE 'WAM-MN9_150D%')

		UNION ALL

		SELECT
			TAG_PREFIX,
			TIME,
			Tank_Type,
			TankVol,
			TankCnt

		FROM (
				 SELECT
					 row_number()
					 OVER (
					 PARTITION BY TAG_PREFIX
						 ORDER BY Time DESC ) AS rk,
					 TAG_PREFIX,
					 TIME,
					 (nvl(TNK_1_WAT_LVL, 0) +
					  nvl(TNK_2_WAT_LVL, 0) +
					  nvl(TNK_3_WAT_LVL, 0) +
					  nvl(TNK_4_WAT_LVL, 0) +
					  nvl(TNK_WAT_1_LVL, 0) +
					  nvl(TNK_WAT_10_LVL, 0) +
					  nvl(TNK_WAT_11_LVL, 0) +
					  nvl(TNK_WAT_2_LVL, 0) +
					  nvl(TNK_WAT_3_LVL, 0) +
					  nvl(TNK_WAT_305A_LVL, 0) +
					  nvl(TNK_WAT_305B_LVL, 0) +
					  nvl(TNK_WAT_305C_LVL, 0) +
					  nvl(TNK_WAT_305D_LVL, 0) +
					  nvl(TNK_WAT_305E_LVL, 0) +
					  nvl(TNK_WAT_310A_LVL, 0) +
					  nvl(TNK_WAT_310B_LVL, 0) +
					  nvl(TNK_WAT_310C_LVL, 0) +
					 nvl(TNK_WAT_310D_LVL, 0) +
					  nvl(TNK_WAT_4_LVL, 0) +
					  nvl(TNK_WAT_6_LVL, 0) +
					  nvl(TNK_WAT_A_LVL, 0) +
					  nvl(TNK_WAT_B_LVL, 0) +
					  nvl(TNK_WAT_C_LVL, 0) +
					  nvl(TNK_WAT_D_LVL, 0)) *
					 20                       AS TankVol,
					 GAS_VC,
					 CASE WHEN (TNK_1_WAT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_2_WAT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_3_WAT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_4_WAT_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_1_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_10_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_11_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_2_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_3_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_305A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_305B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_305C_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_305D_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_305E_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_310A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_310B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_310C_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_310D_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_4_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_6_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_c_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_WAT_d_LVL) >= 0
						 THEN 1
					 ELSE 0 END               AS TankCnt,
					 'WAT'                    AS Tank_Type
				 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
				 WHERE TIME >= ADD_MONTHS(TRUNC(SYSDATE), -12)
			 ) Vol
		WHERE (Vol.TankVol > 0 AND TIME >= ADD_MONTHS(TRUNC(SYSDATE), -12))
		  AND (TAG_PREFIX LIKE 'WAM-CH320C1-160H%'
				OR TAG_PREFIX LIKE 'WAM-CH452K29150H%'
				  OR TAG_PREFIX LIKE 'WAM-CH533B3_80D%'
				OR TAG_PREFIX LIKE 'WAM-CL29_150H%'
				OR TAG_PREFIX LIKE 'WAM-CL29_160H%'
				OR TAG_PREFIX LIKE 'WAM-HP13_150%'
				OR TAG_PREFIX LIKE 'WAM-LM8_115H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_150H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_160H%'
				OR TAG_PREFIX LIKE 'WAM-MN9_150D%')

		UNION ALL

		SELECT
			TAG_PREFIX,
			TIME,
			Tank_Type,
			TankVol,
			TankCnt

		FROM (
				 SELECT
					 row_number()
					 OVER (
						 PARTITION BY TAG_PREFIX
						ORDER BY Time DESC ) AS rk,
					 TAG_PREFIX,
					 TIME,
					 (nvl(TNK_CND_1_LVL, 0) +
					  nvl(TNK_CND_2_LVL, 0) +
					  nvl(TNK_CND_3_LVL, 0) +
					  nvl(TNK_CND_305A_LVL, 0) +
					  nvl(TNK_CND_305B_LVL, 0) +
					  nvl(TNK_CND_305C_LVL, 0) +
					  nvl(TNK_CND_305D_LVL, 0) +
					  nvl(TNK_CND_305E_LVL, 0) +
					  nvl(TNK_CND_305F_LVL, 0) +
					  nvl(TNK_CND_310A_LVL, 0) +
					  nvl(TNK_CND_310B_LVL, 0) +
					  nvl(TNK_CND_310C_LVL, 0) +
					  nvl(TNK_CND_310D_LVL, 0) +
					  nvl(TNK_CND_311_LVL, 0) +
					  nvl(TNK_CND_4_LVL, 0) +
					  nvl(TNK_CND_5_LVL, 0) +
					  nvl(TNK_CND_6_LVL, 0) +
					  nvl(TNK_CND_7_LVL, 0) +
					  nvl(TNK_CND_8_LVL, 0) +
					  nvl(TNK_CND_A_LVL, 0) +
					  nvl(TNK_CND_B_LVL, 0) +
					  nvl(TNK_CND_C_LVL, 0)) *
					 20                       AS TankVol,
					 GAS_VC,
					 CASE WHEN (TNK_CND_1_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_2_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_3_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305C_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305D_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305E_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_305F_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_310A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_310B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_310C_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_310D_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_311_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_4_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_5_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_6_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_7_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_8_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_A_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_B_LVL) >= 0
						 THEN 1
					 ELSE 0 END +
					 CASE WHEN (TNK_CND_C_LVL) >= 0
						 THEN 1
					 ELSE 0 END               AS TankCnt,
					 'CND'                    AS Tank_Type
				 FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
				 Where TIME >= ADD_MONTHS(TRUNC(SYSDATE), -12)
		) Vol
		WHERE (Vol.TankVol > 0)
		  AND (TAG_PREFIX LIKE 'WAM-CH320C1-160H%'
				OR TAG_PREFIX LIKE 'WAM-CH452K29150H%'
				  OR TAG_PREFIX LIKE 'WAM-CH533B3_80D%'
				OR TAG_PREFIX LIKE 'WAM-CL29_150H%'
				OR TAG_PREFIX LIKE 'WAM-CL29_160H%'
				OR TAG_PREFIX LIKE 'WAM-HP13_150%'
				OR TAG_PREFIX LIKE 'WAM-LM8_115H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_150H%'
				OR TAG_PREFIX LIKE 'WAM-ML11_160H%'
				OR TAG_PREFIX LIKE 'WAM-MN9_150D%')
	'''

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
		SELECT RT.runTicketStartDate AS date
			  ,PTD.TAG
			  ,RT.ticketType
			  ,RT.tankCode
			  ,DT.Facilitykey
			  ,RT.grossVolume
		  FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
		  JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
			  ON DW.API = PTD.API
		  JOIN [TeamOptimizationEngineering].[dbo].[DimensionsTanks] AS DT
			ON DT.Facilitykey = DW.Facilitykey
		  JOIN [EDW].[Enbase].[RunTicket] AS RT
			ON RT.tankCode = DT.TankCode
		 WHERE DT.BusinessUnit = 'North'
		   AND CAST(RT.runTicketStartDate AS DATE) >= '01-01-2017'
		   AND DW.API IN ('4903729563', '4903729534', '4903729531',
							 '4903729560', '4903729561', '4903729555',
						  '4903729556', '4903729582', '4903729584',
						  '4903729551', '4900724584', '4903729547',
						  '4903729468', '4903729548', '4903729519',
						  '4903729514');
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

def turbine_pull():
	connection = cx_Oracle.connect("REPORTING", "REPORTING", "L48APPSP1.WORLD")

	cursor = connection.cursor()
	query = ("""
		SELECT  TAG_PREFIX
				,TRUNC(TIME) AS flow_date
				,MAX(CTS_VC) AS volume
		FROM DATA_QUALITY.PI_WAM_ALL_WELLS_OPS
		WHERE (CTS_VC IS NOT NULL)
		  AND (TAG_PREFIX LIKE 'WAM-CH320C1-160H%'
			OR TAG_PREFIX LIKE 'WAM-CH452K29150H%'
			OR TAG_PREFIX LIKE 'WAM-CH533B3_80D%'
			OR TAG_PREFIX LIKE 'WAM-CL29_150H%'
			OR TAG_PREFIX LIKE 'WAM-CL29_160H%'
			OR TAG_PREFIX LIKE 'WAM-HP13_150%'
			OR TAG_PREFIX LIKE 'WAM-LM8_115H%'
			OR TAG_PREFIX LIKE 'WAM-ML11_150H%'
			OR TAG_PREFIX LIKE 'WAM-ML11_160H%'
			OR TAG_PREFIX LIKE 'WAM-MN9_150D%')
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

def tag_pull():
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
				,DW.WellFlac
		FROM [TeamOptimizationEngineering].[Reporting].[PITag_Dict] AS PTD
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsWells] AS DW
			ON PTD.API = DW.API
		JOIN [TeamOptimizationEngineering].[dbo].[DimensionsFacilities] AS DF
			ON DW.Facilitykey = DF.Facilitykey
		GROUP BY PTD.TAG, PTD.API, DF.Facilitykey, DF.FacilityName, DF.FacilityCapacity, DW.WellFlac;
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

def tank_split(df):
	water_df = df[df['tank_type'] == 'WAT'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	water_df.columns = ['tag_prefix', 'time', 'water', 'tankcnt']
	oil_df = df[df['tank_type'] == 'CND'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	oil_df.columns = ['tag_prefix', 'time', 'oil', 'tankcnt']
	total_df = df[df['tank_type'] == 'TOT'][['tag_prefix', 'time', 'tankvol', 'tankcnt']]
	total_df.columns = ['tag_prefix', 'time', 'total', 'tankcnt']

	base_df = water_df.merge(oil_df, on=['tag_prefix', 'time', 'tankcnt'], how='outer')
	df = base_df.merge(total_df, on=['tag_prefix', 'time', 'tankcnt'], how='outer')

	df.loc[df['oil'].isnull(), 'oil'] = df.loc[df['oil'].isnull(), 'total'] - \
										df.loc[df['oil'].isnull(), 'water']

	return df.sort_values(['tag_prefix', 'time'])

def rate(df):
	df.loc[:,'oil_rate'] = np.nan
	df.loc[:,'water_rate'] = np.nan
	df.loc[:,'total_rate'] = np.nan
	for tag in df['tag_prefix'].unique():
		df.loc[df['tag_prefix'] == tag, 'oil_rate'] = \
			   df[df['tag_prefix'] == tag]['oil'] - \
			   df[df['tag_prefix'] == tag]['oil'].shift(1)
		df.loc[df['tag_prefix'] == tag, 'water_rate'] = \
			   df[df['tag_prefix'] == tag]['water'] - \
			   df[df['tag_prefix'] == tag]['water'].shift(1)
		df.loc[df['tag_prefix'] == tag, 'total_rate'] = \
			   df[df['tag_prefix'] == tag]['total'] - \
			   df[df['tag_prefix'] == tag]['total'].shift(1)
		tanks = df[df['tag_prefix'] == tag]['tankcnt'].max()
		df.drop(df[(df['tag_prefix'] == tag) & (df['tankcnt'] < tanks)].index, inplace=True)
	return df.sort_values(['tag_prefix', 'time'])

def outlier_regression(df, tank_type):
	lr = LinearRegression()
	poly = PolynomialFeatures(5)
	x_poly = poly.fit_transform(df['days'].values.reshape(-1, 1))
	lr = lr.fit(x_poly, df[tank_type])
	y = lr.predict(x_poly)
	dev = np.std(abs(df[tank_type] - y))
	if (dev != 0) & (df[(abs(df[tank_type] - y) <= 1.96 * dev)].shape[0] != 0):
		return df[(abs(df[tank_type] - y) <= 1.96 * dev) & \
				  (df[tank_type].notnull())]\
				 [['tag_prefix', 'time', tank_type, 'tankcnt', 'days', 'volume']]
	else:
		return df[['tag_prefix', 'time', tank_type, 'tankcnt', 'days', 'volume']]

def rebuild(df):
	return_df = pd.DataFrame(columns=['TAG_PREFIX', 'DateKey', 'TANK_TYPE', \
									  'TANKLVL', 'predict', 'rate2', \
									  'TANKCNT', 'CalcDate', 'Volume'])

	# Convert DateKey into days since first day
	df.loc[:,'time'] = pd.to_datetime(df['time'])
	day_min = df['time'].min()
	df.loc[:,'days'] = (df['time'] - day_min).dt.total_seconds() / (24 * 60 * 60)

	# Loop through the same model building process for water, oil, and total
	for tank_type in ['oil']:
		if not df[df[tank_type].notnull()].empty:
			full_df = df[df[tank_type].notnull()]

			# Block for removing outliers with linear regression
			# full_df = outlier_regression(full_df, tank_type)

			# Calculate an initial rate from nonnull values
			full_df.loc[:,'rate'] = (full_df[tank_type] - \
								  	 full_df[tank_type].shift(1)) / \
								 	((full_df['time'] - \
								  	 full_df['time'].shift(1)) / \
								  	 np.timedelta64(1, 'h'))

			# Limit dataframe to values where the rate is not 0
			# This is meant to remove repeated values but keep any negative
			# jumps (hauls) which are then reset to a rate of 0
			value_limited_df = full_df[full_df['rate'] != 0]
			value_limited_df.loc[value_limited_df['rate'] < -20, 'rate'] = 0

			# Remove any small negative fluctuations
			rate_limited_df = value_limited_df[value_limited_df['rate'] >= 0]

			# Calculate second rates off of these filtered values
			rate_limited_df.loc[:,'rate2'] = \
					(rate_limited_df[tank_type] - \
						rate_limited_df[tank_type].shift(1)) / \
					((rate_limited_df['time'] - \
						rate_limited_df['time'].shift(1)) / \
						np.timedelta64(1, 'h'))

			# Fill negative rates with values from original rate calculation
			# and forward fill 0 rates
			rate_limited_df.loc[rate_limited_df['rate2'] < 0, 'rate2'] = np.nan
			rate_limited_df['rate2'].fillna(rate_limited_df['rate'], inplace=True)
			rate_limited_df.loc[rate_limited_df['rate2'] == 0, 'rate2'] = np.nan
			rate_limited_df['rate2'].fillna(method='ffill', inplace=True)

			# Limit columns for both dataframes before merging and backfill nan
			full_df = full_df[['tag_prefix', 'time', tank_type, 'tankcnt']]
			rate_limited_df = rate_limited_df[['tag_prefix', 'time', tank_type, 'rate2']]
			type_df = pd.merge(full_df, rate_limited_df, how='left', on=['time', 'tag_prefix', tank_type])
			type_df.fillna(method='bfill', inplace=True)

			# Fill in tank types depending on which iteration we're on
			if tank_type == 'oil':
				type_df.loc[:,'TANK_TYPE'] = np.full(type_df.shape[0], 'CND')
			if tank_type == 'water':
				type_df.loc[:,'TANK_TYPE'] = np.full(type_df.shape[0], 'WAT')
			if tank_type == 'total':
				type_df.loc[:,'TANK_TYPE'] = np.full(type_df.shape[0], 'TOT')

			# Fill in columns to match those expected in SQLDW and append this
			# to the return dataframe
			type_df.rename(index=str, columns={'tag_prefix':'TAG_PREFIX', 'time':'DateKey', \
											   tank_type:'TANKLVL', 'tankcnt':'TANKCNT', \
											   'rate2':'Rate'}, inplace=True)
			type_df.loc[:,'CalcDate'] = type_df['DateKey']
			return_df = return_df.append(type_df)

	return_df = return_df[['TAG_PREFIX', 'DateKey', 'TANK_TYPE', 'TANKLVL', \
						   'Rate', 'TANKCNT', 'CalcDate']]

	return return_df.sort_values(['TAG_PREFIX', 'DateKey'])

def build_loop(df, tic_df):
	r_df = pd.DataFrame(columns=['TAG_PREFIX', 'DateKey', 'TANK_TYPE', \
								 'TANKLVL', 'TANKCNT', 'CalcDate', 'Volume'])
	for tag in df['tag_prefix'].unique():
		ticket = tic_df[(tic_df['ticketType'] != 'Disposition') & (tic_df['TAG'] == tag)]
		if df[(df['tag_prefix'] == tag) & (df['oil'].notnull())].shape[0] == 0:
			pass
		elif not ticket.empty:
			max_date = ticket['date'].max()
			if max_date > df[df['tag_prefix'] == tag]['time'].max():
				max_date = df[df['tag_prefix'] == tag]['time'].max() - pd.Timedelta('1 days')

			rtag_df = rebuild(df[(df['time'] >= max_date + \
							  pd.Timedelta('1 hours')) & \
							 (df['tag_prefix'] == tag)])
			r_df = r_df.append(rtag_df)

		else:
			rtag_df = rebuild(df[df['tag_prefix'] == tag])
			r_df = r_df.append(rtag_df)
	return r_df

def turb_contr(gwr_df, turbine_df):
	gwr_df['DateKey'] = pd.DatetimeIndex(gwr_df['DateKey']).normalize()
	gwr_df = gwr_df.groupby(['FacilityName', 'TANK_TYPE', 'DateKey'], as_index=False).mean()

	turbine_df.loc[:, 'flow_date'] = pd.to_datetime(turbine_df['flow_date'])
	turbine_df.loc[:, 'contr'] = np.nan
	turbine_df.loc[:, 'oil'] = np.nan
	turbine_df = turbine_df[turbine_df['flow_date'] >= gwr_df['DateKey'].min()]

	for fac in turbine_df['FacilityName'].unique():
		for time in turbine_df[turbine_df['FacilityName'] == fac]['flow_date']:
			contr = turbine_df[(turbine_df['FacilityName'] == fac) & \
							   (turbine_df['flow_date'] == time)]['volume'].sum()
			turbine_df.loc[(turbine_df['FacilityName'] == fac) & \
						   (turbine_df['flow_date'] == time), 'contr'] = contr
			match_df = gwr_df[(gwr_df['FacilityName'] == fac) & (gwr_df['DateKey'] == time)]
			print(match_df)
			if len(match_df['Rate'].values) > 0:
				turbine_df.loc[(turbine_df['FacilityName'] == fac) & \
							   (turbine_df['flow_date'] == time), 'oil'] = match_df['Rate'].values[0]

	turbine_df.loc[:, 'oil_contr'] = (turbine_df['oil'] - turbine_df['oil'].shift(1)) * \
									 (turbine_df['volume'] / turbine_df['contr'])
	return turbine_df

def sql_push(df):
	params = urllib.parse.quote_plus('Driver={SQL Server Native Client 11.0};\
									 Server=SQLDW-TEST-L48.BP.Com;\
									 Database=TeamOperationsAnalytics;\
									 trusted_connection=yes'
									 )
	engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

	df.to_sql('cleanGWR', engine, schema='dbo', if_exists='replace', index=False)

def test_plot(df, clean_df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.plot(df['time'], df['oil'], label='GWR Reading')
	ax.plot(clean_df['DateKey'], clean_df['predict'], color='red', label='Cleaned Values')

	cnt = 0
	if len(ax.xaxis.get_ticklabels()) > 12:
		for label in ax.xaxis.get_ticklabels():
			if cnt % 17 == 0:
				label.set_visible(True)
			else:
				label.set_visible(False)
			cnt += 1
	plt.ylim(ymin=0)
	plt.xticks(rotation='vertical')
	plt.xlabel('Date')
	plt.ylabel('bbl Oil')
	plt.title('Cleaned GWR Data for {}'.format(clean_df['TAG_PREFIX'].unique()[0].lstrip('WAM-')))

	plt.savefig('images/new_wells/{}_{}.png'.format(clean_df['TANK_TYPE'].unique()[0], \
												   clean_df['TAG_PREFIX'].unique()[0]))

def rate_plot(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.plot(df['DateKey'], df['rate'], label='Cleaned GWR Rates')

	plt.ylim(ymin=0)
	plt.xticks(rotation='vertical')
	plt.xlabel('Date')
	plt.ylabel('bbl/hr Oil')
	plt.title('GWR Rates for {}'.format(df['TAG_PREFIX'].unique()[0].lstrip('WAM-')))

	plt.savefig('images/new_wells/{}rate_{}.png'.format(df['TANK_TYPE'].unique()[0], \
												   df['TAG_PREFIX'].unique()[0]))


if __name__ == '__main__':
	t0 = time.time()
	df = rate(tank_split(oracle_pull()))
	tic_df = ticket_pull()

	df.to_csv('temp_gwr.csv')
	df = pd.read_csv('temp_gwr.csv')
	tic_df.to_csv('temp_ticket.csv')
	tic_df = pd.read_csv('temp_ticket.csv')

	tic_df['date'] = pd.to_datetime(tic_df['date'])
	df['time'] = pd.to_datetime(df['time'])
	clean_rate_df = build_loop(df, tic_df)
	# sql_push(clean_rate_df)

	turb_df = shift_volumes(turbine_pull())
	tag_df = tag_pull()

	turb_df = pd.merge(turb_df, tag_df, how='left', right_on='tag_prefix', \
					   left_on='tag_prefix')

	facility_rate_df = pd.merge(clean_rate_df, tag_df, how='left', \
								right_on='tag_prefix', left_on='TAG_PREFIX')
	facility_rate_df = facility_rate_df[['FacilityName', 'TANK_TYPE', \
										 'DateKey', 'Rate']].groupby(\
					   ['FacilityName', 'TANK_TYPE', 'DateKey'], \
					   as_index=False).mean()
	facility_rate_df.sort_values(['FacilityName', 'DateKey'], inplace=True)

	contribution_df = turb_contr(facility_rate_df, turb_df)

	t1 = time.time()
	print('Took {} seconds to run.'.format(t1-t0))

	# df = tank_pull()
	# df = pd.read_csv('temp_tank.csv')
	# df = df[df['tag_prefix'] == 'WAM-USANL20-60']

	# o_df = oracle_pull()
	# df = rate(tank_split(o_df))
	# df.to_csv('temp_gwr.csv')
	# df = pd.read_csv('temp_gwr.csv')
	# df.drop('Unnamed: 0', axis=1, inplace=True)

	# ticket_df = ticket_pull()
	# ticket_df.to_csv('temp_ticket.csv')
	# tic_df = pd.read_csv('temp_ticket.csv')
	# tic_df['date'] = pd.to_datetime(tic_df['date'])
	# df['time'] = pd.to_datetime(df['time'])
	# r_df = build_loop(df, tic_df)
	# sql_push(r_df)

	# for tag in r_df['TAG_PREFIX'].unique():
	# 	test_plot(df[(df['tag_prefix'] == tag) & (df['time'] >= '02-01-2018')], \
	# 			  r_df[(r_df['TAG_PREFIX'] == tag) & (r_df['TANK_TYPE'] == 'CND')].sort_values('DateKey'))
	# 	rate_plot(r_df[(r_df['TAG_PREFIX'] == tag) & (r_df['DateKey'] >= '02-01-2018')].sort_values('DateKey'))
