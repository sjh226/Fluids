import pickle
import pandas as pd
import numpy as np

def max_tank(df):
    tank_code = 0
    max_val = 0
    for tank in df['tankCode'].unique():
        if df[df['tankCode'] == tank].shape[0] > max_val:
            tank_code = tank
            max_val = df[df['tankCode'] == tank].shape[0]
    return df[df['tankCode'] == tank_code]


if __name__ == '__main__':
    enb_df = pickle.load(open('data/enb.pkl', 'rb'))
    enb_df.drop_duplicates(inplace=True)
    ticket_df = pickle.load(open('data/ticket.pkl', 'rb'))
    ticket_df.drop_duplicates(inplace=True)

    tank_df = max_tank(enb_df)
