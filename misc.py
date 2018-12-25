import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# misc functions
def check_nan(df):
    num_idx = 0
    last_nan_idx = 0
    for idx, row in df.iterrows():
        num_idx += 1
        if np.isnan(np.min(row.values)):
            last_nan_idx += 1
    # print('Total idx', num_idx, 'Last nan idx', last_nan_idx, 'Remain idx', num_idx - last_nan_idx)
    return df[last_nan_idx:]


def normalize(df):
    factors_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]
    normalized_factor_df = (factors_df - factors_df.min()) / (factors_df.max() - factors_df.min())

    return pd.concat(objs=[normalized_factor_df, y_df], axis=1, join='inner')


def lag_data(df, lag):
    factors_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]
    if lag <= 0:
        return df
    temp = []
    for i in range(1, lag + 1):
        temp.append(factors_df.shift(i))
    return pd.concat(objs=[pd.concat(objs=temp, axis=1, join='inner'), y_df], axis=1, join='inner')[lag:]


def create_test_feeds(X, Y, shift=5, size=30, tol=30):
    i = -1
    lo_idx = 0
    temp = []
    feeds = []
    t = []# for verification
    while lo_idx < len(X):
        hi_idx = lo_idx + size
        if len(X[lo_idx:hi_idx]) >= tol:
            temp.append((X[lo_idx:hi_idx], Y[lo_idx:hi_idx],))
            if i == -1:
                feeds.append((X[0:size], Y[0:size]))
                t.append(X[lo_idx:hi_idx])
            else:
                feeds.append((X[size + i * shift: size + (i + 1) * shift], Y[size + i * shift: size + (i + 1) * shift]))
                t.append(X[size + i * shift: size + (i + 1) * shift])
            i += 1

        elif lo_idx == 0:
            feeds.append((X[lo_idx:hi_idx], Y[lo_idx:hi_idx]))
            temp.append(X[lo_idx:hi_idx])
            t.append(X[lo_idx:hi_idx])
            # print len(X[lo_idx:hi_idx])
        lo_idx += shift
    #     print('t len',len(t))
    recon_ = np.concatenate(tuple(t), axis=0)
    print('#Rolling test windows', len(temp))
    # verification
    if np.array_equal(a1=recon_, a2=X[:len(recon_)]):
        return temp, feeds


def rolling_window(X, Y, shift=50, size=400, tol=200):
    lo_idx = 0
    temp = []
    while lo_idx < len(X):
        hi_idx = lo_idx + size
        if len(X[lo_idx:hi_idx]) >= tol:
            temp.append((X[lo_idx:hi_idx], Y[lo_idx:hi_idx],))
            # print len(X[lo_idx:hi_idx])
        elif lo_idx == 0:
            temp.append((X[lo_idx:hi_idx], Y[lo_idx:hi_idx],))
        lo_idx += shift
    print('#Rolling windows', len(temp))
    return temp

def query_range(df, lo, hi, lag):
    df2 = df.query('code >= %d and code <= %d' %(lo, hi)).dropna(how='all', axis='columns').replace([np.inf, -np.inf],
                                                                                        np.nan).interpolate()
    # print(df2.head(20))
    df2 = check_nan(df2)
    if df2.index.size < 200:
        return
    df2 = normalize(df2)
    df2 = lag_data(df2, lag)
    # print(df2.describe())
    return df2

def query_data(df, query_code, lag=3):
    # print('Code:', query_code, 'Lag', lag)
    df2 = df.query('code == %d' % query_code).dropna(how='all', axis='columns').replace([np.inf, -np.inf],
                                                                                        np.nan).interpolate()
    # print(df2.head(20))
    df2 = check_nan(df2)
    if df2.index.size < 200:
        return
    df2 = normalize(df2)
    df2 = lag_data(df2, lag)
    # print(df2.describe())
    return df2


def my_pearsonr(x, y):
    #     if len(x) > 1 and len(y)>1
    # squeeze dims
    x = np.squeeze(x)
    y = np.squeeze(y)
    return pearsonr(x, y)