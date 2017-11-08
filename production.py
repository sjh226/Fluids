import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


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

			# WHERE P.Wellkey = 2745

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

def iqr_outlier(df):
	df_out = df
	for value in ['Gas']:
		median = np.median(df_out[value])
		iqr_25_50 = stats.iqr(df_out[value], rng=(25, 50))
		iqr_50_75 = stats.iqr(df_out[value], rng=(50, 75))
		lower_bound = median - iqr_25_50
		upper_bound = median + iqr_50_75
		df_out = df_out[(df_out[value] > lower_bound) & (df_out[value] < upper_bound)]
	return df_out

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

	# Train test split
	arima_df = lgr_df.set_index('date')
	train, test = arima_df[:-10], arima_df[-10:]

	# Can we grid search these parameters?
	# (9, 1, 7) -> 0.0455929 RMSE
	arima_model = ARIMA(train, order=(9, 1, 1))
	model_fit = arima_model.fit(disp=0)
	# print(model_fit.summary())
	pred = model_fit.predict()
	forecast, std_error, conf_int  = model_fit.forecast(steps=len(test))
	error = mean_squared_error(test, forecast)
	r_2 = r2_score(test, forecast)
	print('RMSE of forecast in %\n---------------')
	print(np.sqrt(error), '\n')
	print('R^2 of forecast\n---------------')
	print(r_2)

	print(np.mean(pred))
	# print(ARIMA.score(model_fit))

	if plot == True:
		plt.close()
		fig, ax = plt.subplots(1,1,figsize=(20,10))

		ax.plot(lgr_df['date'].values, lgr_df['Oil'].values, 'k-', label='True LGR')
		ax.plot(lgr_df['date'].values[1:-10], pred, 'r-', label='Predicted LGR')
		# ax.plot(lgr_df['date'].values, lgr_df['lgr'].values, 'k-', label='True LGR')
		# ax.plot(lgr_df['date'].values, pred, 'b-', label='Predicted LGR')

		ax.set_xlabel('Date')
		ax.set_ylabel('Liquid to Gas Ratio')

		plt.legend()
		plt.title('LGR on Well {}'.format(df['WellFlac'].unique()[0]))
		plt.savefig('figures/lgr_{}.png'.format(df['WellFlac'].unique()[0]))

def arima_params(df):
	plt.close()
	autocorrelation_plot(df)
	plt.show()


if __name__ == '__main__':
	df = prod_query()

	# Try limiting on a single variable
	lim_df = iqr_outlier(df)
	lgr(lim_df, plot=True)

	# for well in df['WellFlac'].unique()[:20]:
	# 	lgr(df[df['WellFlac'] == well], plot=True)
	# arima_params(df[['DateKey', 'lgr']].values)
	# prod_plot(df)

	# df_lift = lift_query()
	# Need to parse out when status = 'Down - Gas Lift'
