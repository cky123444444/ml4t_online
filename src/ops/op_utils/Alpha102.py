from src.utils.logger import setup_logger
import numpy as np
import pandas as pd

logger = setup_logger("Alpha102")
from numpy import abs
from numpy import log
from numpy import sign
from .helper import *
import time
import json
import os
from joblib import Parallel, delayed
from scipy.stats import linregress
import pywt



def compute_alpha(name, method, window):
    if name in {'alpha041', 'alpha054', 'alpha101',  'alpha106',  'alpha119', 'alpha121',  'alpha124'}:
        # print(f'generating {name}')
        return f'{name}_new', method()
    elif name in {'alpha104', 'alpha129', 'alpha131'}:
        # print(f'generating {name}')
        return f'{name}_{window[0]}_{window[1]}_new', method(window[0], window[1])
    else:
        # print(f'generating {name}_{window}')
        return f'{name}_{window}_new', method(window)


def get_ts_alpha(df):
    start_time = time.time()
    stock = Alphas(df)
    alpha_tasks = [
        ('alpha006', stock.alpha006, 60), ('alpha006', stock.alpha006, 240),
        ('alpha007', stock.alpha007, 60), ('alpha007', stock.alpha007, 240),
        ('alpha009', stock.alpha009, 60), ('alpha009', stock.alpha009, 240),
        ('alpha010', stock.alpha010, 60), ('alpha010', stock.alpha010, 240),
        ('alpha012', stock.alpha012, 60), ('alpha012', stock.alpha012, 240),
        ('alpha013', stock.alpha013, 30), ('alpha013', stock.alpha013, 40), ('alpha013', stock.alpha013, 60),
        ('alpha014', stock.alpha014, 30), ('alpha014', stock.alpha014, 40), ('alpha014', stock.alpha014, 60),
        ('alpha021', stock.alpha021, 60), ('alpha021', stock.alpha021, 240),
        ('alpha023', stock.alpha023, 60), ('alpha023', stock.alpha023, 240),
        ('alpha024', stock.alpha024, 60), ('alpha024', stock.alpha024, 240),
        ('alpha026', stock.alpha026, 60), ('alpha026', stock.alpha026, 240),
        ('alpha028', stock.alpha028, 60), ('alpha028', stock.alpha028, 240),
        ('alpha035', stock.alpha035, 60), ('alpha035', stock.alpha035, 240),
        ('alpha041', stock.alpha041, None),
        ('alpha043', stock.alpha043, 60), ('alpha043', stock.alpha043, 240),
        ('alpha046', stock.alpha046, 60), ('alpha046', stock.alpha046, 240),
        ('alpha049', stock.alpha049, 60), ('alpha049', stock.alpha049, 240),
        ('alpha051', stock.alpha051, 60), ('alpha051', stock.alpha051, 240),
        ('alpha053', stock.alpha053, 60), ('alpha053', stock.alpha053, 240),
        ('alpha054', stock.alpha054, None),
        ('alpha084', stock.alpha084, 60), ('alpha084', stock.alpha084, 240),
        ('alpha101', stock.alpha101, None),
        ('alpha102', stock.alpha102, 60), ('alpha102', stock.alpha102, 240),
        ('alpha103', stock.alpha103, 60), ('alpha103', stock.alpha103, 240),
        ('alpha104', stock.alpha104, [60, 240]), ('alpha104', stock.alpha104, [240, 720]),
        ('alpha105', stock.alpha105, 60), ('alpha105', stock.alpha105, 240),
        ('alpha106', stock.alpha106, None),
        ('alpha107', stock.alpha107, 60), ('alpha107', stock.alpha107, 240),
        ('alpha108', stock.alpha108, 60), ('alpha108', stock.alpha108, 240),
        ('alpha109', stock.alpha109, 60), ('alpha109', stock.alpha109, 240),
        ('alpha110', stock.alpha110, 60), ('alpha110', stock.alpha110, 240),
        ('alpha111', stock.alpha111, 60), ('alpha111', stock.alpha111, 240),
        ('alpha112', stock.alpha112, 20), ('alpha112', stock.alpha112, 60), ('alpha112', stock.alpha112, 240),
        ('alpha113', stock.alpha113, 60), ('alpha113', stock.alpha113, 240),
        ('alpha115', stock.alpha115, 60), ('alpha115', stock.alpha115, 240),
        ('alpha116', stock.alpha116, 60), ('alpha116', stock.alpha116, 240),
        ('alpha117', stock.alpha117, 60), ('alpha117', stock.alpha117, 240),
        ('alpha118', stock.alpha118, 60), ('alpha118', stock.alpha118, 240),
        ('alpha119', stock.alpha119, None),
        ('alpha120', stock.alpha120, 60), ('alpha120', stock.alpha120, 240),
        ('alpha121', stock.alpha121, None),
        ('alpha122', stock.alpha122, 60), ('alpha122', stock.alpha122, 240),
        ('alpha123', stock.alpha123, 60), ('alpha123', stock.alpha123, 240),
        ('alpha124', stock.alpha124, None),
        ('alpha125', stock.alpha125, 60), ('alpha125', stock.alpha125, 240),
        ('alpha126', stock.alpha126, 60), ('alpha126', stock.alpha126, 240),
        ('alpha127', stock.alpha127, 60), ('alpha127', stock.alpha127, 240),
        ('alpha128', stock.alpha128, 60), ('alpha128', stock.alpha128, 240),
        ('alpha129', stock.alpha129, [60, 0.00556]), ('alpha129', stock.alpha129, [240, 0.00333]),
        ('alpha130', stock.alpha130, 60), ('alpha130', stock.alpha130, 240),
        ('alpha131', stock.alpha131, [60, 1]), ('alpha131', stock.alpha131, [240, 1]),
        ('alpha131', stock.alpha131, [60, 2]), ('alpha131', stock.alpha131, [240, 2]),
    ]

    results = Parallel(n_jobs=-1)(
        delayed(compute_alpha)(name, method, window) for name, method, window in alpha_tasks
    )
    # for name, values in results:
    #     df[name] = values
    #把所有 alpha 特征先组成一个新的 DataFrame
    # for name, values in results:
        # print(f"{name}: {type(values)}")
    new_features = pd.concat(
        [pd.DataFrame({name: values}) for name, values in results],
        axis=1
    )

    # 合并到原 df 上（按列拼接）
    df = pd.concat([df, new_features], axis=1)
    end_time = time.time()
    logger.info(f"运行时间: {end_time - start_time:.4f} 秒")
    return df

    
def get_ts_alpha_config(df, config_path='alpha102_short.json'):
    start_time = time.time()
    stock = Alphas(df)
    
    # 如果是相对路径（不包含目录分隔符），自动拼接到 op_utils 目录
    if not os.path.isabs(config_path) and os.sep not in config_path and '/' not in config_path:
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    logger.info(f"Using alpha config path: {config_path}")
    # 确保文件存在
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Alpha config file not found: {config_path}")
    
    # 从 JSON 配置文件加载任务
    with open(config_path, 'r', encoding='utf-8') as file:
        alpha_config = json.load(file)

    alpha_tasks = []

    for alpha_name, window_list in alpha_config.items():
        method = getattr(stock, alpha_name, None)
        if method:
            for window in window_list:
                alpha_tasks.append((alpha_name, method, window))
        else:
            logger.warning(f"{alpha_name} method not found in Alphas class.")

    # dump alpha_tasks into debug files
    # debug_path = os.path.join("app/debug_output", "alpha_tasks_debug.json")

    logger.info(f"Generating {len(alpha_tasks)} alpha features...")
    # threading backend avoids pickle serialization of the Alphas object;
    # pandas/numpy ops release the GIL so threads run in true parallel.
    try:
        results = Parallel(n_jobs=6)(
            delayed(compute_alpha)(name, method, window) for name, method, window in alpha_tasks
        )
    except PermissionError:
        # Some restricted environments disallow loky semaphore creation.
        logger.warning("Parallel execution unavailable, fallback to sequential alpha generation.")
        results = [compute_alpha(name, method, window) for name, method, window in alpha_tasks]

    # 只有在有结果时才进行特征拼接
    if results:
        new_features = pd.concat(
            [pd.DataFrame({name: values}) for name, values in results],
            axis=1
        )
        # 合并到原 df 上（按列拼接）
        df = pd.concat([df, new_features], axis=1)
    else:
        logger.warning("No alpha features generated (empty configuration).")
    
    end_time = time.time()
    logger.info(f"运行时间: {end_time - start_time:.4f} 秒")
    return df


class Alphas(object):
    def __init__(self, df_data):

        self.open = df_data['open'] 
        self.high = df_data['high'] 
        self.low = df_data['low']   
        self.close = df_data['close'] 
        self.volume = df_data['volume']
        self.returns = df_data['returns'] 
        #self.vwap = (df_data['S_DQ_AMOUNT']*1000)/(df_data['S_DQ_VOLUME']*100+1) 
        self.vwap = df_data['vwap']
        #self.market_cap = df_data['market_cap']
        
    
    
    # Alpha#6	 (-1 * correlation(open, volume, 10))
    def alpha006(self, window=10):
        alpha = -1 * correlation(self.open, self.volume, window)
        return alpha.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#7	 ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha007(self, window=60):
        adv20 = sma(self.volume, window)
        alpha = -1 * ts_rank(abs(delta(self.close, window//3)), window) * sign(delta(self.close, window//3))
        alpha[adv20 >= self.volume] = -1
        return alpha
    
    
    # Alpha#9	 ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self,window):
        delta_close = delta(self.close, window)
        cond_1 = ts_min(delta_close, 5*window) > 0
        cond_2 = ts_max(delta_close, 5*window) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#10	 rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self,window=1):
        delta_close = delta(self.close, window)
        cond_1 = ts_min(delta_close, 4*window) > 0
        cond_2 = ts_max(delta_close, 4*window) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    
    # Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self,window=1):
        return sign(delta(self.volume, window)) * (-1 * delta(self.close, window))


    # Alpha#13	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha013(self,window=1):
        return ((sma(self.close, 7*window) / 7*window) - self.close)
    
    # Alpha#14 migrate from alpha032
    def alpha014(self, window=1):
        return correlation(self.vwap, delay(self.close, 5*window),230*window)

    # Alpha#21	 ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self,window=2):
        cond_1 = sma(self.close, 4*window) + stddev(self.close, 4*window) < sma(self.close, window)
        cond_2 = sma(self.volume, 10*window) / self.volume < 1
        #alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index)
        alpha = pd.Series(1.0, index=self.close.index)
#        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
#                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha
    

    # Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha023(self, window=2):
        cond = sma(self.high, 10*window) < self.high
        delta_high = delta(self.high, window).fillna(0)
        alpha = pd.Series(0.0, index=self.close.index)
        alpha[cond] = -1.0 * delta_high[cond]
        #alpha = pd.DataFrame(np.zeros_like(self.close),index=self.close.index,columns=['close'])
        #alpha.loc[cond,'close'] = -1 * delta(self.high, window).fillna(value=0)
        return alpha
    
    # Alpha#24	 ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self, window=3):
        cond = delta(sma(self.close, 33*window), 33*window) / delay(self.close, 33*window) <= 0.05
        alpha = -1 * delta(self.close, 33)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 33*window))
        return alpha
    

    
    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self, window=1):
        df = correlation(ts_rank(self.volume, 5*window), ts_rank(self.high, 5*window), 5*window)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3*window)
    
    
    # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self,window=5):
        adv20 = sma(self.volume, 4*window)
        df = correlation(adv20, self.low, window)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((df + ((self.high + self.low) / 2)) - self.close)
    
    
    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self,window=16):
        return ((ts_rank(self.volume, 2*window) *
                 (1 - ts_rank(self.close + self.high - self.low, window))) *
                (1 - ts_rank(self.returns, 2*window)))
            

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low),0.5) - self.vwap
    
        
    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self, window=1):
        adv20 = sma(self.volume, 20*window)
        return ts_rank(self.volume / adv20, 20*window) * ts_rank((-1 * delta(self.close, 7*window)), 8*window)

    
    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self, window=10):
        inner = ((delay(self.close, 2*window) - delay(self.close, window)) / 10) - ((delay(self.close, window) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

     
    
    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self, window=10):
        inner = (((delay(self.close, 2*window) - delay(self.close, window)) / 10) - ((delay(self.close, window) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha
    
    
    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self, window=10):
        inner = (((delay(self.close, 2*window) - delay(self.close, window)) / 10) - ((delay(self.close, window) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha
    
        
    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self, window=9):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), window)

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    
    
    # Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    def alpha084(self,window=1):
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15*window)), 21*window), delta(self.close,5*window)/1000.0)
    
     

    # Alpha#101	 ((close - open) / ((high - low) + .001))
    def alpha101(self):
        return (self.close - self.open) /((self.high - self.low) + 0.001)

    # RSI
    def alpha102(self, window=14):
        delta = self.close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_up = sma(up, window)
        avg_down = sma(down, window)
        return 1 - 1/(1+(avg_up/avg_down))
    
    # Price Momentum
    def alpha103(self, window=7):
        return self.close.pct_change(window)

    # Trend Strength
    def alpha104(self, short=9, long=21):
        ma_short = sma(self.close, short)
        ma_long = sma(self.close, long)
        signal = pd.Series(0, index=self.close.index)
        signal[(ma_short > ma_long) & (ma_short.shift() <= ma_long.shift())] = 1
        signal[(ma_short < ma_long) & (ma_short.shift() >= ma_long.shift())] = 0
        return signal

    # price momentum volume adjustment
    def alpha105(self, window=12):
        total_return = self.close.pct_change(window)
        var = self.returns.rolling(window).std()
        return total_return/var
    
    #price accelaration
    def alpha106(self):
       return self.close.diff().diff()
   
    #window percent off high
    def alpha107(self, window=168):
       window_high = ts_max(self.close, window)
       return (self.close - window_high)/window_high
   
    # ---------------------------------------------------------------------
    # Alpha 108 – 12-day rolling close std
    # ---------------------------------------------------------------------
    def alpha108(self, window=12): 
        return self.close.pct_change().rolling(window).std()
    
    # ---------------------------------------------------------------------
    # Alpha 109 – 20-day rolling Sharpe ratio (price momentum adjusted by risk)
    # ---------------------------------------------------------------------
    def alpha109(self, window: int = 20, eps: float = 1e-6):
        """Rolling Sharpe ratio.

        To prevent extreme values when volatility approaches zero, we explicitly
        set the factor to 0 whenever the sampled standard-deviation is < ``eps``.
        """
        mean_ret = sma(self.returns, window)
        vol = stddev(self.returns, window)

        sharpe = pd.Series(0.0, index=self.returns.index)
        mask = vol > eps
        sharpe.loc[mask] = mean_ret.loc[mask] / vol.loc[mask]

        return sharpe.replace([np.inf, -np.inf], 0).fillna(0)

    # ---------------------------------------------------------------------
    # Alpha 110 – 20-day correlation between price and volume (price-volume sentiment)
    # ---------------------------------------------------------------------
    def alpha110(self, window: int = 20):
        corr = correlation(self.close, self.volume, window)
        return corr.replace([np.inf, -np.inf], 0).fillna(0)
    
    # ---------------------------------------------------------------------
    # Alpha 111 – 30-day skewness of returns (captures asymmetry of distribution)
    # ---------------------------------------------------------------------
    def alpha111(self, window: int = 30):
        skew = self.returns.rolling(window).skew()
        return skew.fillna(0)

    # ---------------------------------------------------------------------
    # Alpha 112 – Bollinger %B (position of price inside 20-day Bollinger bands)
    # ---------------------------------------------------------------------
    def alpha112(self, window: int = 20, num_std: float = 2.0):
        mid = sma(self.close, window)
        sd = stddev(self.close, window)
        upper = mid + num_std * sd
        lower = mid - num_std * sd
        percent_b = (self.close - lower) / (upper - lower)
        return percent_b.replace([np.inf, -np.inf], 0).fillna(0)
    
    # ---------------------------------------------------------------------
    # Alpha 113 – Relative volume-adjusted return (RVOL * price change)
    # ---------------------------------------------------------------------
    def alpha113(self, window: int = 20):
        adv = sma(self.volume, window)
        rvol = self.volume / (adv + 1e-9)
        pct_change = self.close.pct_change().fillna(0)
        return (rvol * pct_change).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ---------------------------------------------------------------------
    # Alpha 114 – Size-adjusted return (small-cap bias factor)
    # ---------------------------------------------------------------------
    def alpha114(self):
        inv_mcap = 1 / (self.market_cap + 1e-9)
        return (self.returns * inv_mcap).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ---------------------------------------------------------------------
    # Alpha 115 – Turnover Volatility
    # ---------------------------------------------------------------------
    def alpha115(self, window=12):
        turnover = self.volume * self.close
        return turnover.rolling(window).std()
    
    # ---------------------------------------------------------------------
    # Alpha 116 – low-close or open range
    # ---------------------------------------------------------------------
    def alpha116(self, window=24):
        window_min = np.minimum(self.open, self.close).rolling(window).min()
        return self.low/ window_min - 1
    
    # ---------------------------------------------------------------------
    # Alpha 117 – Close Location Value
    # ---------------------------------------------------------------------
    def alpha117(self, window=24):
        ts_low = ts_min(self.high, window)
        ts_high = ts_max(self.low, window) 
        return ((self.close - ts_low) - (ts_high - self.close))/ (ts_high - ts_low + 1e-6)
    
    
    # ---------------------------------------------------------------------
    # Alpha 118 – Gap
    # ---------------------------------------------------------------------
    def alpha118(self, window=24):
        first_open = self.open.rolling(window).apply(lambda x: x.iloc[0], raw=False)
        return self.close - first_open
    
    # ---------------------------------------------------------------------
    # Alpha 119 – consecutive up/down window
    # ---------------------------------------------------------------------
    def alpha119(self):
        dir = np.sign(self.returns)
        group = (dir != dir.shift(1)).cumsum()
        streak  =dir.groupby(group).cumcount()+1
        return streak * dir
    
    # ---------------------------------------------------------------------
    # Alpha 120 – Volume Spike
    # ---------------------------------------------------------------------
    def alpha120(self, window=20):
        return (self.volume - sma(self.volume, window)) / stddev(self.volume, window)
    
    # ---------------------------------------------------------------------
    # Alpha 121 – On-Balance Volume
    # ---------------------------------------------------------------------
    def alpha121(self):
        dir = np.where(self.returns > 0, 1, -1)
        return (dir * self.volume).cumsum()
    
    # ---------------------------------------------------------------------
    # Alpha 122 – sentiment_divergence
    # ---------------------------------------------------------------------
    def alpha122(self, window=24):
        return self.close.pct_change(window)-self.volume.pct_change(window)
    
    # ---------------------------------------------------------------------
    # Alpha 123 – window volatility
    # ---------------------------------------------------------------------
    def alpha123(self, window=24):
        ts_low = ts_min(self.high, window)
        ts_high = ts_max(self.low, window) 
        return (ts_high - ts_low)/self.close
    
    # ---------------------------------------------------------------------
    # Alpha 123 – vwap bias
    # ---------------------------------------------------------------------
    def alpha124(self):
        return (self.close - self.vwap)/self.vwap
    
    
    # ---------------------------------------------------------------------
    # Alpha 124 – william index
    # ---------------------------------------------------------------------
    def alpha125(self, window=24):
        ts_high = ts_max(self.low, window) 
        ts_low = ts_min(self.high, window)
        return (ts_high - self.close) / (ts_high - ts_low) * -100
    
    # ---------------------------------------------------------------------
    # Alpha 125 – money flow index
    # ---------------------------------------------------------------------
    def alpha126(self, window=24):
        typical_price = (self.high + self.low + self.close) /3
        raw_money_flow = typical_price * self.volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        mfr = positive_flow.rolling(window).sum() / (negative_flow.rolling(window).sum() + 1e-6)
        return 100 - 100 / (1 + mfr)
    
    # ---------------------------------------------------------------------
    # Alpha 126 – freq domin argmax
    # ---------------------------------------------------------------------
    def alpha127(self, window=30):
        return self.close.rolling(window).apply(
            lambda x: np.fft.fftfreq(window, 1)[:window//2][np.argmax(fourier_power(x)[1:]) + 1],
            raw=True)

    # ---------------------------------------------------------------------
    # Alpha 127 – spectrum slope
    # ---------------------------------------------------------------------
    def alpha128(self, window=60):
        # power = self.close.rolling(window).apply(fourier_power)
        # freq = np.fft.fftfreq(window,1)
        # valid = freq > 0
        # freq = freq[valid]
        # power = power[valid]
        # log_freq, log_power = log(freq), log(power)
        # return -linregress(log_freq, log_power).slope
        def spectral_slope(x):
            power = fourier_power(x)
            freq = np.fft.fftfreq(len(x), 1)[:len(x)//2]
            
            # 取正频率部分
            valid = freq > 0
            freq = freq[valid]
            power = power[valid]
            
            # 防止log(0)错误
            power = np.where(power == 0, 1e-10, power)
            
            log_freq = log(freq)
            log_power = log(power)
            
            slope, _, _, _, _ = linregress(log_freq, log_power)
            return -slope  # 取负值作为因子

        return self.close.rolling(window).apply(spectral_slope, raw=True)

    
    # ---------------------------------------------------------------------
    # Alpha 128 – hf power vs. all_power
    # ---------------------------------------------------------------------
    def alpha129(self, window=60, thres=0.3):
        # power, freq = fourier_power(self.close, window)
        # hf_power = np.sum(power[freq > thres])
        # total_power = np.sum(power)
        # return hf_power / total_power if total_power > 0 else 0
        print(window)
        print(thres)
        def hf_power_ratio(x):
            power = fourier_power(x)
            n = len(x)
            freq = np.fft.fftfreq(n, 1)[:n//2]
            hf_power = np.sum(power[freq > thres])
            total_power = np.sum(power)
            return hf_power / total_power if total_power > 0 else 0
    
        return self.close.rolling(window).apply(hf_power_ratio, raw=True)
    
    # ---------------------------------------------------------------------
    # Alpha 129 – shannon entropy
    # ---------------------------------------------------------------------
    def alpha130(self, window=60):
        # power, freq = fourier_power(self.close, window)
        # power_sum = np.sum(power)
        # if power_sum == 0:
        #     return 0.0

        # p = power / power_sum
        # p = np.clip(p, 1e-12, 1)  # 避免 log(0)

        # # Shannon 熵
        # return -np.sum(p * np.log(p))/np.log(len(p))
        def spectral_entropy(x):
            power = fourier_power(x)  # 返回单边频谱功率
            power_sum = np.sum(power)
            if power_sum == 0:
                return 0.0
            p = power / power_sum
            p = np.clip(p, 1e-12, 1)  # 避免 log(0)
            shannon_entropy = -np.sum(p * np.log(p)) / np.log(len(p))
            return shannon_entropy

        return self.close.rolling(window).apply(spectral_entropy, raw=True)
    

    # ---------------------------------------------------------------------
    # Alpha 130 – wavelet momentum
    # ---------------------------------------------------------------------
    def alpha131(self, window=60, level=1):
        # coeffs = pywt.wavedec(self.close.rolling(window), wavelet='db4', level=4)
        # cD = coeffs[level]  # 第 i 层细节系数（越高 i 越低频）
        # return np.mean(cD)
        def wavelet_detail_mean(x):
            coeffs = pywt.wavedec(x, wavelet='db4', level=4)
            cD = coeffs[level]  # 第 level 层的细节系数
            return np.mean(cD)

        return self.close.rolling(window).apply(wavelet_detail_mean, raw=True)
    
