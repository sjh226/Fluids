import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def max_tank(df):
	tank_code = 0
	max_val = 0
	for tank in df['tankCode'].unique():
		# if tank in df1['tankCode'].unique():
		if df[df['tankCode'] == tank].shape[0] > max_val:
			tank_code = tank
			# print(max_val)
			max_val = df[df['tankCode'] == tank].shape[0]
	return df[df['tankCode'] == tank_code]

def water_comp(df):
	return df[df['newWaterInventory'].notnull()]

def ticket_org(df):
	return df.sort_values('runTicketStartDate')

def ticket_plot(df, data_source):
	plt.close()
	if data_source == 'ticket':
		label = 'Open Ticket'
		title = 'Open Measurement'
		save = 'open_meas'
	elif data_source == 'enb':
		label = 'Liquid Gauge'
		title = 'Gauge Measurement'
		save = 'gauge'
	fig, ax = plt.subplots(1,1,figsize=(20,10))
	# meads = df['openMeasFeet'].values + ((df['openMeasInches'].values + df['openMeasFract'].values) / 12)
	ax.plot(df['Date'].values, df['meas'], 'k-', label=label)
	# ax.plot(df_g[df_g['createdDate'] > np.min(df['runTicketStartDate'])].values, df_g['meas'], 'k-', label='Gauge Measurement')
	ax.plot(df['Date'].values, df['tankHeight'].values / 12, 'r--', label='Max Tank Height')
	if data_source == 'ticket':
		ax.plot(df['Date'].values[::20], df['rate'].values[::20], 'b--', label='Measurement Rate')
	elif data_source == 'enb':
		ax.plot(df['Date'].values, df['rate'].values, 'b--', label='Measurement Rate')
	ax.set_xlabel('Date')
	ax.set_ylabel('Open Measurement (feet)')
	plt.legend()
	plt.title('{} on Tank {}'.format(title, df['tankCode'].unique()[0]))
	plt.savefig('figures/{}.png'.format(save))

def feat_eng(df, source):
	if source == 'ticket':
		df['Date'] = pd.to_datetime(df['runTicketStartDate'].values)
		df['meas'] = df['openMeasFeet'].values + ((df['openMeasInches'].values + df['openMeasFract'].values) / 12)
		df['rate'] = df['meas'] - df['meas'].shift(-1)
		df['weights'] = (np.mean(df[df['Date'] > df['Date'] - \
									 datetime.timedelta(days=30)]['meas']) * .7) + \
						(np.mean(df[df['Date'] <= df['Date'] - \
									 datetime.timedelta(days=30)]['meas']) * .3)
	elif source == 'enb':
		df['Date'] = pd.to_datetime(df['createdDate'].values)
		df['meas'] = (df['liquidGaugeFeet'].values + ((df['liquidGaugeInches'].values + df['liquidGaugeQuarter'].values * 4) / 12)).astype(float)
		df['rate'] = df['meas'] - df['meas'].shift(-1)
		df['tankHeight'] = df['Height']

	return df


if __name__ == '__main__':
	enb_df = pickle.load(open('data/enb.pkl', 'rb'))
	enb_df.drop_duplicates(inplace=True)
	ticket_df = pickle.load(open('data/ticket.pkl', 'rb'))
	ticket_df.drop_duplicates(inplace=True)

	tank_df = max_tank(enb_df)
	# tank_df = water_comp(tank_df)
	tank_df = tank_df.sort_values('createdDate')
	tank_df = tank_df.fillna(0)

	tank_ticket = max_tank(ticket_df)
	# tank_ticket = ticket_df[ticket_df['tankCode'] == tank_df['tankCode'].unique()[0]]
	tank_ticket = ticket_org(tank_ticket)
	tank_ticket = tank_ticket.fillna(method='backfill')

	# tank_df = enb_df[enb_df['tankCode'] == tank_ticket['tankCode'].unique()[0]]

	df1 = feat_eng(tank_ticket, 'ticket')
	df2 = feat_eng(tank_df, 'enb')
	ticket_plot(df1, 'ticket')
	ticket_plot(df2, 'enb')
