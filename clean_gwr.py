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
			 --Where TIME >= trunc(sysdate-2)
			 ) Vol
		WHERE Vol.TankVol > 0
			  --AND TIME >= trunc(sysdate)
			  --AND rk = 1

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
				 --WHERE TIME >= trunc(sysdate - 2)
			 ) Vol
		WHERE Vol.TankVol > 0
			  --AND TIME >= trunc(sysdate)
			  --AND rk = 1

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
				  --Where TIME >= trunc(sysdate-2)
		) Vol
		WHERE Vol.TankVol > 0
			  --AND TIME >= trunc(sysdate)
			  --AND rk = 1
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
        SELECT CAST(RT.runTicketStartDate AS DATE) AS date
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
			   df[df['tag_prefix'] == tag]['oil'].shift(-2) - \
			   df[df['tag_prefix'] == tag]['oil'].shift(2)
		df.loc[df['tag_prefix'] == tag, 'water_rate'] = \
			   df[df['tag_prefix'] == tag]['water'] - \
			   df[df['tag_prefix'] == tag]['water'].shift(1)
		df.loc[df['tag_prefix'] == tag, 'total_rate'] = \
			   df[df['tag_prefix'] == tag]['total'] - \
			   df[df['tag_prefix'] == tag]['total'].shift(1)
		tanks = df[df['tag_prefix'] == tag]['tankcnt'].max()
		df.drop(df[(df['tag_prefix'] == tag) & (df['tankcnt'] < tanks)].index, inplace=True)
	return df.sort_values(['tag_prefix', 'time'])

def rebuild(df):
	return_df = pd.DataFrame(columns=['TAG_PREFIX', 'DateKey', 'TANK_TYPE', \
									  'TANKLVL', 'TANKCNT', 'CalcDate'])

	# Convert DateKey into days since first day
	df.loc[:,'time'] = pd.to_datetime(df['time'])
	day_min = df['time'].min()
	df.loc[:,'days'] = (df['time'] - day_min).dt.total_seconds() / (24 * 60 * 60)

	# Loop through the same model building process for water, oil, and total
	if not df[df['water'].notnull()].empty:
		# Remove null values for model building
		w_df = df[df['water'].notnull()]
		# Build a linear regression with X-degree polynomial (currently 3 works best)
		w_lr = LinearRegression()
		w_poly = PolynomialFeatures(3)
		w_x_poly = w_poly.fit_transform(w_df['days'].values.reshape(-1, 1))
		w_lr = w_lr.fit(w_x_poly, w_df['water'])
		w_y = w_lr.predict(w_x_poly)
		# Calculate standard deviation and remove values outside of a 95% CI
		w_dev = np.std(abs(w_df['water'] - w_y))
		water_df = w_df[(abs(w_df['water'] - w_y) <= 1.28 * w_dev) & \
						(w_df['water'].notnull())][['tag_prefix', 'time', 'water', 'tankcnt']]
		water_df.loc[:,'TANK_TYPE'] = np.full(water_df.shape[0], 'WAT')
		# Format columns to match that in SQL Server
		water_df.rename(index=str, columns={'tag_prefix':'TAG_PREFIX', 'time':'DateKey', \
											'water':'TANKLVL', 'tankcnt':'TANKCNT'}, \
											inplace=True)
		water_df.loc[:,'CalcDate'] = water_df['DateKey']
		return_df = return_df.append(water_df)

	if not df[df['oil'].notnull()].empty:
		o_df = df[df['oil'].notnull()]
		o_lr = LinearRegression()
		o_poly = PolynomialFeatures(3)
		o_x_poly = o_poly.fit_transform(o_df['days'].values.reshape(-1, 1))
		o_lr = o_lr.fit(o_x_poly, o_df['oil'])
		o_y = o_lr.predict(o_x_poly)
		o_dev = np.std(abs(o_df['oil'] - o_y))
		oil_df = o_df[(abs(o_df['oil'] - o_y) <= 1.28 * o_dev) & \
					  (o_df['oil'].notnull())][['tag_prefix', 'time', 'oil', 'tankcnt']]
		oil_df.loc[:,'TANK_TYPE'] = np.full(oil_df.shape[0], 'CND')
		oil_df.rename(index=str, columns={'tag_prefix':'TAG_PREFIX', 'time':'DateKey', \
										  'oil':'TANKLVL', 'tankcnt':'TANKCNT'}, \
										  inplace=True)
		oil_df.loc[:,'CalcDate'] = oil_df['DateKey']
		return_df = return_df.append(oil_df)

	if not df[df['total'].notnull()].empty:
		t_df = df[df['total'].notnull()]
		t_lr = LinearRegression()
		t_poly = PolynomialFeatures(3)
		t_x_poly = t_poly.fit_transform(t_df['days'].values.reshape(-1, 1))
		t_lr = t_lr.fit(t_x_poly, t_df['total'])
		t_y = t_lr.predict(t_x_poly)
		t_dev = np.std(abs(t_df['total'] - t_y))
		total_df = t_df[(abs(t_df['total'] - t_y) <= 1.28 * t_dev) & \
					  (t_df['total'].notnull())][['tag_prefix', 'time', 'total', 'tankcnt']]
		total_df.loc[:,'TANK_TYPE'] = np.full(total_df.shape[0], 'TOT')
		total_df.rename(index=str, columns={'tag_prefix':'TAG_PREFIX', 'time':'DateKey', \
											'total':'TANKLVL', 'tankcnt':'TANKCNT'}, \
											inplace=True)
		total_df.loc[:,'CalcDate'] = total_df['DateKey']
		return_df = return_df.append(total_df)

	return_df = return_df[['TAG_PREFIX', 'DateKey', 'TANK_TYPE', 'TANKLVL', 'TANKCNT', 'CalcDate']]

	return return_df.sort_values(['TAG_PREFIX', 'DateKey'])

def sql_push(df):
	params = urllib.parse.quote_plus('Driver={SQL Server Native Client 11.0};\
									 Server=SQLDW-TEST-L48.BP.Com;\
									 Database=TeamOperationsAnalytics;\
									 trusted_connection=yes'
									 )
	engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params)

	# # Code to try speeding up SQL insert
	# conn = engine.connect().connection
	# cursor = conn.cursor()
	# records = [tuple(x) for x in df.values]
	#
	# insert_ = """
	# 	INSERT INTO dbo.CleanGWR
	# 	(TAG_PREFIX
	# 	,DateKey
	# 	,TANK_TYPE
	# 	,TANKLVL
	# 	,TANKCNT
	# 	,CalcDate)
	# 	VALUES"""
	# def chunker(seq, size):
	# 	return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
	# for batch in chunker(records, 1000):
	#     rows = ','.join(batch)
	#     insert_rows = insert_ + rows
	#     cursor.execute(insert_rows)
	#     conn.commit()

	test = df.iloc[:200]

	test.to_sql('cleanGWR', engine, schema='dbo', if_exists='replace', index=False)

def test_plot(df, clean_df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	ax.plot(df['time'], df['total'], label='GWR Reading')
	ax.plot(clean_df['DateKey'], clean_df['TANKLVL'], color='red', label='Cleaned Values')

	cnt = 0
	if len(ax.xaxis.get_ticklabels()) > 12:
		for label in ax.xaxis.get_ticklabels():
			if cnt % 17 == 0:
				label.set_visible(True)
			else:
				label.set_visible(False)
			cnt += 1

	plt.xticks(rotation='vertical')
	plt.title('Cleaned GWR Data for {}'.format(clean_df['TAG_PREFIX'].unique()[0].lstrip('WAM-')))

	plt.savefig('images/gwr/test/{}_{}.png'.format(clean_df['TANK_TYPE'].unique()[0], \
												   clean_df['TAG_PREFIX'].unique()[0]))


if __name__ == '__main__':
	# o_df = oracle_pull()
	# df = rate(tank_split(o_df))
	# df.to_csv('temp_gwr.csv')
	df = pd.read_csv('temp_gwr.csv')
	df.drop('Unnamed: 0', axis=1, inplace=True)
	lim_df = df

	# ticket_df = ticket_pull()
	# ticket_df.to_csv('temp_ticket.csv')
	tic_df = pd.read_csv('temp_ticket.csv')
	tic_df['date'] = pd.to_datetime(tic_df['date'])
	lim_df['time'] = pd.to_datetime(lim_df['time'])
	# tic_df = ticket_df[ticket_df['TAG'] == 'WAM-5MILE33-60D']

	r_df = pd.DataFrame()
	for tag in lim_df['tag_prefix'].unique()[:30]:
		ticket = tic_df[(tic_df['ticketType'] != 'Disposition') & (tic_df['TAG'] == tag)]
		if lim_df[(lim_df['tag_prefix'] == tag) & (lim_df['total'].notnull())].shape[0] == 0:
			pass
		elif not ticket.empty:
			max_date = ticket['date'].max().normalize()
			if max_date >= lim_df['time'].max().normalize():
				max_date -= pd.Timedelta('2 days')
			if not lim_df[(lim_df['time'] >= max_date) & (lim_df['tag_prefix'] == tag)].empty:
				rtag_df = rebuild(lim_df[(lim_df['time'] >= max_date) & \
							     (lim_df['tag_prefix'] == tag)])
							    # + pd.Timedelta('1 days'))
			else:
				pass
		else:
			rtag_df = rebuild(lim_df[lim_df['tag_prefix'] == tag])
		r_df = r_df.append(rtag_df)
	# sql_push(df)

	for tag in r_df['TAG_PREFIX'].unique():
		# rate_plot(r_df[(r_df['TAG_PREFIX'] == tag) & (r_df['TANK_TYPE'] == 'CND')])
		test_plot(lim_df[lim_df['tag_prefix'] == tag], \
				  r_df[(r_df['TAG_PREFIX'] == tag) & (r_df['TANK_TYPE'] == 'TOT')].sort_values('DateKey'))
