import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
from statsmodels.tsa.stattools import adfuller



def plot_acf_pacf(df, column_name, lags=40):
    plt.rcParams["figure.figsize"] = 18, 5

    _, axes = plt.subplots(1, 2)

    sgt.plot_acf(df[column_name], zero=False, lags=lags, ax=axes[0])
    sgt.plot_pacf(df[column_name], zero=False, lags=lags, ax=axes[1])

    plt.show()

def calculate_adf_test(df, column_name):
    result = adfuller(df[column_name])
    # Mostrar los resultados con sus respectivos nombres
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def smoother(df, window=100, column="FP_mean"):
    rolling_mean = df[column].rolling(window).mean()
    rolling_var = df[column].rolling(window).var()
    return rolling_mean, rolling_var

def plot_rolling_mean_and_rolling_var(df, window=100, rolling_mean_title="", rolling_variance_title="", column="FP_mean", window_size=30,lags=100):
    df.plot(y=column, use_index=True)
    plt.ylabel(column)
    plt.title(f"Plot of the column '{column}'")
    plt.show()

    plot_acf_pacf(df, column_name=column, lags=40)

    rolling_mean, rolling_var = smoother(df, window=window, column=column)

    _, (ax1, ax2) = plt.subplots(1, 2)
    plt.rcParams["figure.figsize"] = 18, 12

    ax1.plot(rolling_mean)
    if rolling_mean_title:
        ax1.set_title(rolling_mean_title)
    else:
        ax1.set_title(f"Rolling Mean (Window = {window_size})")

    ax2.plot(rolling_var)
    if rolling_variance_title:
        ax2.set_title(rolling_variance_title)
    else:
        ax2.set_title(f"Rolling Variance (Window = {window_size})")

    plt.show()

    calculate_adf_test(df,column_name=column)