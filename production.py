import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

	df['Date'] = pd.to_datetime(df['DateKey'])

	return df.drop_duplicates()

def oil_well(df):
    max_rows = 0
    max_well = None
    for well in df['WellFlac'].unique():
        row_count = df[pd.notnull(df[df['Well1_WellFlac'] == well]['Oil'])].shape[0]
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
	ax.plot(df['Date'].values, df['Water'], 'b-', label='Water')

	ax.set_xlabel('Date')
	ax.set_ylabel('MCF (log scale)')
	ax.set_yscale('log')

	plt.legend()
	plt.title('Production on Well {}'.format(df['WellFlac'].unique()[0]))
	plt.savefig('figures/production.png')


if __name__ == '__main__':
	# df = prod_query()
    oil_df = oil_well(df)
	# prod_plot(oil_df)

    df_lift = lift_query()
    # Need to parse out when status = 'Down - Gas Lift'
