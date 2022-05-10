# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
from statistics import mean

# import matplotlib as plt
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from datetime import datetime
#
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

def extract_data():
    df = pd.read_csv( 'store-sales-time-series-forecasting/transactions.csv')
    print(df)

    # ##CONVERT TIMESTAMP TO INT FOR TRAIN
    # df['@timestamp'] = df['@timestamp'].map(lambda x:  int(round(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f').timestamp())), na_action=None)
    # df['timestamp_min_max'] = (df['@timestamp'] - df['@timestamp'].min()) / (
    # df['@timestamp'].max() - df['@timestamp'].min())
    #
    # ###CHANGE STAKE FEATURE TO LOG BASE 10
    # df['stake'] = df['stake'].map(lambda x:  math.log(x, 10) if x != 0 else 0, na_action=None)
    #
    # ### REMOVE 0 FOR CALCULATION OF GROUP VARIANCE INVESTIGATION (prevents errors)
    # df['stake_no_zero'] = df['stake'].loc[~(df['stake'] == 0)]
    # df['pblocks_1h_no_zero'] = df['pblocks_1h'].loc[~(df['pblocks_1h'] == 0)]
    # df['wblocks_1h_no_zero'] = df['wblocks_1h'].loc[~(df['wblocks_1h'] == 0)]
    # df['max_s1_1h_no_zero'] = df['max_s1_1h'].loc[~(df['max_s1_1h'] == 0)]
    #
    #
    # df['stake_binary'] = df['stake'].map(lambda x:  0 if x == 0 else 1, na_action=None)
    # df['status'] = df['status'].map(lambda x: 1 if x == 'Online' else 0, na_action=None)
    #
    # ###ORDER USERS BY STAKE RANK
    # address_order = df[['stake', 'address']].groupby(by=["address"], dropna=False).apply(
    #     lambda x: x.mean()).sort_values('stake').index
    #
    #
    # sorterIndex = dict(zip(address_order, range(len(address_order))))
    # df['stake_rank'] = df['address'].map(sorterIndex)
    #
    #
    # # STAKE RANK NORMILZATION FEATURE
    # df['stake_rank_minmax'] = (df['stake_rank'] - df['stake_rank'].min()) / (
    # df['stake_rank'].max() - df['stake_rank'].min())
    # return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    extract_data()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
