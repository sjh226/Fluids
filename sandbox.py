import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    meas = df['openMeasFeet'].values + ((df['openMeasInches'].values + df['openMeasFract'].values) / 12)
    plt.plot(pd.to_datetime(df['runTicketStartDate'].values), meas, 'k-')
    plt.plot(pd.to_datetime(df['runTicketStartDate'].values), df['tankHeight'].values / 12, 'r--')
    plt.xlabel('Date')
    plt.ylabel('Open Measurement (feet)')
    plt.title('Open Measurement on Tank {}'.format(df['tankCode'].unique()))
    plt.savefig('figures/open_meas.png')


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

    ticket_plot(tank_ticket)
