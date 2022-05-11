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
from datetime import datetime
from pathlib import Path


#
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
#
# def create_timestamp():
#     df_train = pd.read_csv( 'store-sales-time-series-forecasting/train.csv')
#     df_train['timestamp'] = df_train['date'].map(lambda x:  int(round(datetime.strptime(x, '%Y-%m-%d').timestamp())), na_action=None)
#     filepath = Path('store-sales-time-series-forecasting/train_timestamp.csv')
#     df_train.to_csv(filepath)


def extract_data():
    df_transactions = pd.read_csv( 'store-sales-time-series-forecasting/transactions.csv')
    df_train = pd.read_csv( 'store-sales-time-series-forecasting/train_timestamp.csv')
    df_sample_submission = pd.read_csv( 'store-sales-time-series-forecasting/sample_submission.csv')
    df_test = pd.read_csv( 'store-sales-time-series-forecasting/test.csv')


    # create_features:
    # -sales yesterday
    # -sales last_week
    # -sales_ last_month
    # -sales_last_year

    return df_train, df_sample_submission, df_test

def explore_data(df):
    df_store_1 = df.loc[(df['store_nbr'] == 2) & (df['timestamp'] == 1405051200)]
    # df_store_1 = df.loc[(df['timestamp'] == 1405224000)]
    # df_store_1 = df.loc[(df['store_nbr'] == 3)]

    ids = df_store_1['id']
    dates = df_store_1['timestamp']
    stores = df_store_1['timestamp']

    ids_all = df['id']
    stores_all = df['store_nbr']
    dates_all = df['timestamp']
    family_all = df['family']

    print(len(ids.value_counts()))
    print(len(ids))

    print("ALL")
    print(len(ids_all.value_counts()))
    print(len(dates_all.value_counts()))
    print(len(stores_all.value_counts()))
    print(len(family_all.value_counts()))

    print("****")

    print((dates_all.value_counts()))



    print(len(dates.value_counts()))
    print((dates.value_counts()))
    print(len(dates))


def explore_sample_submission(df):
    ids = df['id']
    print(len(ids.value_counts()))
    print(len(ids))


def explore_date_data(df):
    print(df[['timestamp', 'date']].sort_values('date'))





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create_timestamp()
    df_train, df_sample_submission, df_test = extract_data()
    # explore_data(df_train)
    # print("***")
    # explore_sample_submission(df_sample_submission)
    explore_date_data(df_train)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
