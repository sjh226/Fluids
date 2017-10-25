import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def max_tank(df):
	tank_code = 0
	max_val = 0
	for tank in df['tankCode'].unique():
		if df[df['tankCode'] == tank].shape[0] > max_val:
			tank_code = tank
			max_val = df[df['tankCode'] == tank].shape[0]
	return df[df['tankCode'] == tank_code]

def water_comp(df):
	return df[df['newWaterInventory'].notnull()]

def ticket_org(df):
	return df.sort_values('runTicketStartDate')

def ticket_plot(df):
	plt.close()
	fig, ax = plt.subplots(1,1,figsize=(20,10))
	# meads = df['openMeasFeet'].values + ((df['openMeasInches'].values + df['openMeasFract'].values) / 12)
	ax.plot(df['runTicketStartDate'].values, df['meas'], 'k-', label='Open Measurement')
	ax.plot(df['runTicketStartDate'].values, df['tankHeight'].values / 12, 'r--', label='Max Tank Height')
	ax.plot(df['runTicketStartDate'].values[::20], df['rate'].values[::20], 'b--', label='Measurement Rate')
	ax.set_xlabel('Date')
	ax.set_ylabel('Open Measurement (feet)')
	plt.legend()
	plt.title('Open Measurement on Tank {}'.format(df['tankCode'].unique()[0]))
	plt.savefig('figures/open_meas.png')

def feat_eng(df, source):
	if source == 'ticket':
		df['runTicketStartDate'] = pd.to_datetime(df['runTicketStartDate'].values)
		df['meas'] = df['openMeasFeet'].values + ((df['openMeasInches'].values + df['openMeasFract'].values) / 12)
		df['rate'] = df['meas'] - df['meas'].shift(-1)
		df['weights'] = (np.mean(df[df['runTicketStartDate'] > df['runTicketStartDate'] - \
									 datetime.timedelta(days=30)]['meas']) * .7) + \
						(np.mean(df[df['runTicketStartDate'] <= df['runTicketStartDate'] - \
									 datetime.timedelta(days=30)]['meas']) * .3)
	elif source == 'enb':
		pass

	return df


if __name__ == '__main__':
	# enb_df = pickle.load(open('data/enb.pkl', 'rb'))
	# enb_df.drop_duplicates(inplace=True)
	ticket_df = pickle.load(open('data/ticket.pkl', 'rb'))
	ticket_df.drop_duplicates(inplace=True)

	# tank_df = max_tank(enb_df)
	# tank_df = water_comp(tank_df)

	tank_ticket = max_tank(ticket_df)
	tank_ticket = ticket_org(tank_ticket)
	tank_ticket = tank_ticket.fillna(method='backfill')

	df = feat_eng(tank_ticket, 'ticket')

	ticket_plot(df)
