import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA


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
			ON P.Wellkey = W.Wellkey
		WHERE P.Wellkey = 2745;
	""")

	# (SELECT TOP 1 MP.Wellkey
	# FROM [OperationsDataMart].[Facts].[Production] AS MP
	# WHERE MP.Oil > 1 AND MP.Water
	# GROUP BY MP.Wellkey
	# ORDER BY COUNT(MP.Oil) DESC)

	cursor.execute(SQLCommand)
	results = cursor.fetchall()

	df = pd.DataFrame.from_records(results)
	connection.close()

	try:
		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
	except:
		df = None
		print('Dataframe is empty')

	df['Date'] = pd.to_datetime(df['DateKey'])

	return df.drop_duplicates()

def oil_well(df):
	max_rows = 0
	max_well = None
	for well in df['WellFlac'].unique():
		print(well)
		row_count = df[(df['WellFlac'] == well) & pd.notnull(df['Oil'])].shape[0]
		if row_count > max_rows:
			max_rows = row_count
			max_well = well
	df_out = df[df['WellFlac'] == max_well]
	return df_out

def lift_query():
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
			  ,DS.Status
			  ,DS.DateKey
		FROM [OperationsDataMart].[Facts].[Production] AS P
		JOIN [OperationsDataMart].[Dimensions].[Wells] AS W
			ON P.Wellkey = W.Wellkey
		JOIN [OperationsDataMart].[Stage].[DowntimeStatus] AS DS
			ON DS.WellFlac = W.WellFlac;
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

	df['Date'] = pd.to_datetime(df['DateKey'])

	return df.drop_duplicates()

def prod_plot(df):
	plt.close()

	fig, ax = plt.subplots(1,1,figsize=(20,10))

	ax.plot(df['Date'].values, df['Gas'].values, 'k-', label='Gas')
	ax.plot(df['Date'].values, df['Oil'].values, 'r-', label='Oil')
	ax.plot(df['Date'].values, df['Water'].values, 'b-', label='Water')
	ax.plot(df['Date'].values, df['lgr'].values, 'g--', label='LGR')

	ax.set_xlabel('Date')
	ax.set_ylabel('MCF (log scale)')
	ax.set_yscale('log')

	plt.legend()
	plt.title('Production on Well {}'.format(df['WellFlac'].unique()[0]))
	plt.savefig('figures/lgr.png')

def lgr(df, plot=False):
	# Look at gas production
	# Create a model to predict the LGR over time
	# Input gas production and LGR prediction to create oil prediction
	# Add in artificial lift analysis (enter a 1 for when a change occurs and
	# created a "dummy" average for days since changed)

	df['lgr'] = (df['Oil'] + df['Water']) / df['Gas']
	df['date'] = pd.to_datetime(df['DateKey'])

	lgr_df = df[['date', 'lgr']]
	lgr_df.replace(np.inf, np.nan, inplace=True)
	lgr_df.dropna(inplace=True)

	arima_df = lgr_df.set_index('date')

	arima_model = ARIMA(arima_df, order=(0,0,5))
	model_fit = arima_model.fit()
	# print(model_fit.summary())
	pred = model_fit.predict(start=np.min(lgr_df['date']), end=np.max(lgr_df['date']))
	forecast = model_fit.forecast()

	if plot == True:
		plt.close()
		fig, ax = plt.subplots(1,1,figsize=(20,10))

		ax.plot(lgr_df['date'].values, lgr_df['lgr'].values, 'k-', label='True LGR')
		ax.plot(lgr_df['date'].values, pred, 'b-', label='Predicted LGR')
		ax.plot

		ax.set_xlabel('Date')
		ax.set_ylabel('Liquid to Gas Ratio')

		plt.legend()
		plt.title('LGR on Well {}'.format(df['WellFlac'].unique()[0]))
		plt.savefig('figures/lgr_pred.png')

def arima_params(df):
	plt.close()
	autocorrelation_plot(df)
	plt.show()


if __name__ == '__main__':
	df = prod_query()
	# oil_df = oil_well(df)
	lgr(df, plot=True)
	# arima_params(df[['DateKey', 'lgr']].values)
	# prod_plot(df)

	# df_lift = lift_query()
	# Need to parse out when status = 'Down - Gas Lift'
