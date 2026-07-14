import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import json
# Lazy import for optional dependency to avoid import errors at module load time
try:
    from binance.client import Client
except Exception:
    Client = None

def z_score_from_json(df, json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)

    z_data = {}  # 用字典暂存所有列的 z-score 结果

    for col in df.columns:
        if col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            if std != 0:
                z_data[col] = (df[col] - mean) / std
            else:
                print(f"⚠️ 列 {col} 的 std 为 0，跳过标准化。")
                z_data[col] = df[col]
        else:
            print(f"⚠️ 列 {col} 不在 JSON 中，跳过。")
            z_data[col] = df[col]

    # 一次性创建 DataFrame
    z_df = pd.concat(z_data, axis=1)
    z_df.index = df.index  # 保留原 index

    return z_df

def rolling_z_score_vol(df, vol_window=60, z_window=10000, exclude_cols=None):
    df_vol_z = df.copy()
    exclude_cols = exclude_cols or []
    result_list = {}

    for col in df.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 计算该列本身的波动率
        returns = df[col].pct_change().fillna(0)
        volatility = returns.rolling(window=vol_window, min_periods=1).std()

        # 波动率调整
        vol_adjusted_col = df[col] / volatility.replace(0, np.nan).fillna(1)

        # 计算滚动均值和滚动标准差（前 window 个点，不含当前点）
        rolling_mean = vol_adjusted_col.shift(1).rolling(window=z_window, min_periods=1).mean()
        rolling_std = vol_adjusted_col.shift(1).rolling(window=z_window, min_periods=1).std()

        # 存储最后一个点的均值和 std
        result_list[col] = {
            'mean': float(rolling_mean.iloc[-1]),
            'std': float(rolling_std.iloc[-1])
        }

        # 避免除以0
        df_vol_z[col] = (vol_adjusted_col - rolling_mean) / rolling_std.replace(0, np.nan)
        df_vol_z[col] = df_vol_z[col].fillna(0)

    return df_vol_z, result_list

def rolling_z_score(df, window=10000, exclude_cols=None):
    """
    对 df 中除 exclude_cols 外的所有列进行滚动窗口的 z-score 归一化。

    每个值减去前 window 个数据的均值，除以前 window 个数据的标准差。

    参数:
        df: pandas.DataFrame，原始数据
        window: int，滚动窗口大小
        exclude_cols: list[str]，不需要归一化的列名列表（可为 None）

    返回:
        pandas.DataFrame，归一化后的数据（原 df 的副本）
    """
    df_z = df.copy()
    exclude_cols = exclude_cols or []
    result_list = {}

    for col in df.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 计算滚动均值和滚动标准差（前 window 个点，不含当前点）
        rolling_mean = df[col].shift(1).rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].shift(1).rolling(window=window, min_periods=1).std()

        # 存储最后一个点的均值和 std（可选）
        result_list[col] = {
            'mean': float(rolling_mean.iloc[-1]),
            'std': float(rolling_std.iloc[-1])
        }

        # 避免除以0
        df_z[col] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        df_z[col] = df_z[col].fillna(0)  # 可根据需要换成原值或 NaN

    return df_z, result_list

#返回单边频谱和对应频率，频率单位为数据粒度的倒数
def fourier_power(df):
    fft_res = np.fft.fft(df)
    n = len(df)
    power = np.abs(fft_res)
    # 计算频谱
    return power[:n//2]

def fourier_freq(df):
    n = len(df)
    freq = np.fft.fftfreq(n,1)
    return freq[:n//2]

# region Auxiliary functions
def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    
    return df.rolling(window).sum()

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()

def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)

def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :] 
    na_series = df.to_numpy()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['close'])  


def norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def minus_date(cur_date, days):
    given_date = datetime.strptime(cur_date, '%Y-%m-%d')
    # 计算前一天日期
    previous_date = given_date - timedelta(days=days)

    # 将前一天日期转换为字符串
    previous_date_string = previous_date.strftime('%Y-%m-%d')
    return previous_date_string

def plus_date(cur_date, days):
    given_date = datetime.strptime(cur_date, '%Y-%m-%d')
    # 计算前一天日期
    previous_date = given_date + timedelta(days=days)

    # 将前一天日期转换为字符串
    previous_date_string = previous_date.strftime('%Y-%m-%d')
    return previous_date_string