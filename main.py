import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

trades = pd.read_csv('CBA_trades.csv')
trades.Time = pd.to_datetime(trades.Time)
trades.set_index('Time', inplace=True)
trades.plot()
plt.show()

mask = (trades.index > dt.datetime(2021, 6, 10, 10, 0, 0)) & (trades.index <= dt.datetime(2021, 6, 10, 16, 0, 0))
trades_mh = trades.iloc[mask]

time_bars = trades_mh.groupby(pd.Grouper(freq='1min')).agg({'Price $': 'ohlc', 'Volume': 'sum'})
time_bars_price = time_bars.loc[:, 'Price $']
time_bars = np.log(time_bars_price.close / time_bars_price.close.shift(1)).dropna()
bin_len = 0.001
plt.hist(time_bars, bins=np.arange(min(time_bars), max(time_bars) + bin_len, bin_len))
plt.show()


def bar(x, y):
    return np.int64(x / y) * y


transactions = 75
tick_bars = trades_mh.groupby(bar(np.arange(len(trades_mh)), transactions)).agg({'Price $': 'ohlc', 'Volume': 'sum'})
tick_bars_price = tick_bars.loc[:, 'Price $']
tick_bars = np.log(tick_bars_price.close / tick_bars_price.close.shift(1)).dropna()
bin_len = 0.0001
plt.hist(tick_bars, bins=np.arange(min(tick_bars), max(tick_bars) + bin_len, bin_len))
plt.show()

traded_volume = 10000
volume_bars = trades_mh.groupby(bar(np.cumsum(trades_mh['Volume']), traded_volume)).agg(
    {'Price $': 'ohlc', 'Volume': 'sum'})
volume_bars_price = volume_bars.loc[:, 'Price $']

volume_bars = np.log(volume_bars_price.close / volume_bars_price.close.shift(1)).dropna()
bin_len = 0.0001
plt.hist(volume_bars, bins=np.arange(min(volume_bars), max(volume_bars) + bin_len, bin_len))
plt.show()

market_value = 700000
dollar_bars = trades_mh.groupby(bar(np.cumsum(trades_mh['Value $']), market_value)).agg(
    {'Price $': 'ohlc', 'Volume': 'sum'})
dollar_bars_price = dollar_bars.loc[:, 'Price $']

dollar_bars = np.log(dollar_bars_price.close / dollar_bars_price.close.shift(1)).dropna()
bin_len = 0.0001
plt.hist(dollar_bars, bins=np.arange(min(dollar_bars), max(dollar_bars) + bin_len, bin_len))
plt.show()

from scipy import stats
import seaborn as sns

cdmx_edad = np.random.normal(0, 20, 10000) + 10
ed_sup_edad = dollar_bars
dollar_bars = np.log(dollar_bars_price.close / dollar_bars_price.close.shift(1)).dropna
volume_bars = np.log(volume_bars_price.close / volume_bars_price.close.shift(1)).dropna
tick_bars = np.log(tick_bars_price.close / tick_bars_price.close.shift(1)).dropna
time_bars = np.log(time_bars_price.close / time_bars_price.close.shift(1)).dropna
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
# bins = np.arange10,61,1
bin_len = 0.001
ax1.hist(time_bars, bins=np.arange(min(time_bars), max(time_bars) + bin_len, bin_len), alpha=0.4, label='Time Bars')
bin_len = 0.0001
ax1.hist(tick_bars, bins=np.arange(min(tick_bars), max(tick_bars) + bin_len, bin_len), alpha=0.4, label='Tick Bars')
ax1.hist(volume_bars, bins=np.arange(min(volume_bars), max(volume_bars) + bin_len, bin_len), alpha=0.4,
         label='Volume Bars')
ax1.hist(dollar_bars, bins=np.arange(min(dollar_bars), max(dollar_bars) + bin_len, bin_len), alpha=0.4,
         label='Dollar Bars')
ax1.legend()
dollar_bars_kde = stats.gaussian_kde(dollar_bars)
tick_bars_kde = stats.gaussian_kde(tick_bars)
volume_bars_kde = stats.gaussian_kde(volume_bars)
time_bars_kde = stats.gaussian_kde(time_bars)
x = np.linspace(-0.001, 0.001, 500)
dollar_bars_curve = dollar_bars_kde(x) * dollar_bars.shape[0]
tick_bars_curve = tick_bars_kde(x) * tick_bars.shape[0]
volume_bars_curve = volume_bars_kde(x) * volume_bars.shape[0]
time_bars_curve = time_bars_kde(x) * time_bars.shape[0]
# ax2.plot𝑥,𝑐𝑑𝑚𝑥𝑐𝑢𝑟𝑣𝑒,𝑐𝑜𝑙𝑜𝑟=′𝑟′
ax2.fill_between(x, 0, time_bars_curve, alpha=1, label='Time Bars')
ax2.fill_between(x, 0, tick_bars_curve, alpha=0.4, label='Tick Bars')
ax2.fill_between(x, 0, volume_bars_curve, alpha=0.4, label='Volume Bars')
ax2.fill_between(x, 0, dollar_bars_curve, alpha=0.4, label='Dollar Bars')
ax1.set_xlim(-0.0015, 0.0015)
# ax2.plot𝑥,𝑒𝑑𝑠𝑢𝑝𝑐𝑢𝑟𝑣𝑒,𝑐𝑜𝑙𝑜𝑟=′𝑏′
ax2.legend()
plt.show()
len_tick_bars = np.arange(min(tick_bars), max(tick_bars) + bin_len, bin_len)
len(len_tick_bars)

sns.kdeplot(dollar_bars, gridsize=1000, shade=True, label='Dollar Bars')
sns.kdeplot(volume_bars, gridsize=1000, shade=True, label='Volume Bars')
sns.kdeplot(tick_bars, gridsize=25, shade=True, label='Tick Bars')
sns.kdeplot(time_bars, gridsize=50, shade=True, label='Time Bars')
plt.xlim(-0.0025, 0.0025)
plt.xlabel('Log Returns ')
plt.ylabel('Frequency')
plt.title('KDE of Standard Price & Volume Bars')
# dollar_bars = np.log𝑑𝑜𝑙𝑙𝑎𝑟𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒/𝑑𝑜𝑙𝑙𝑎𝑟𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒.𝑠ℎ𝑖𝑓𝑡(1).dropna
# volume_bars = np.log𝑣𝑜𝑙𝑢𝑚𝑒𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒/𝑣𝑜𝑙𝑢𝑚𝑒𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒.𝑠ℎ𝑖𝑓𝑡(1).dropna
# tick_bars = np.log𝑡𝑖𝑐𝑘𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒/𝑡𝑖𝑐𝑘𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒.𝑠ℎ𝑖𝑓𝑡(1).dropna
# time_bars = np.log𝑡𝑖𝑚𝑒𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒/𝑡𝑖𝑚𝑒𝑏𝑎𝑟𝑠𝑝𝑟𝑖𝑐𝑒.𝑐𝑙𝑜𝑠𝑒.𝑠ℎ𝑖𝑓𝑡(1).dropna
# sns.kdeplot𝑒𝑑𝑢𝑎𝑐𝑖𝑜𝑛𝑠𝑢𝑝𝑒𝑟𝑖𝑜𝑟[′𝐸𝐷𝐴𝐷′],𝑠ℎ𝑎𝑑𝑒=𝑇𝑟𝑢𝑒
