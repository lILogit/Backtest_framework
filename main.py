"pozor počet pips desetiných míst dopad na a-leves a signal.values"

# pip install bokeh == 3.3.0.dev2 problem with plot
# or pip uninstall bokeh  Step 2 pip install bokeh==2.4.3
import sqlite3
import sys

from matplotlib import pyplot as plt

"""
úkoly:
- update each evening automaticaly
- filter close close and min d 

- kontrola digits!
pondělí se otevírá market už 10h v nedeli 
problem tickformatter
p1.xaxis.formatter=DatetimeTickFormatter(formats=dict(
    days=["%m/%d %H:%M"],
    months=["%m/%d %H:%M"],
    hours=["%m/%d %H:%M"],
    minutes=["%m/%d %H:%M"]
))
"""
from datetime import timedelta
import holidays
import math
import time
import seaborn as sns
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import timezone, datetime
from loguru import logger
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
from bokeh.plotting import output_file, figure, show, ColumnDataSource, curdoc, save
from bokeh.layouts import row, column
from bokeh.models import PreText

mp.set_start_method('fork')  # multiprocessing

# data
# db_name = "/Users/jirka/PycharmProjects/MYProject/mydb.db"
# name_index = "EURUSD_M1_XTB_2023"intr
local_time_zone = 'Europe/Prague'
database = "/Users/jirka/Documents/forex_v1.db"
maximize_field = 'Equity Final [$]'  # "# Trades" #'Equity Final [$]'
time_frame = 60  # seconds
sample_size = 2880  # minutes 43200 = month M1 2880 = 2 days xAPI

item_name = "EURUSD"
scale = 300
min_trades = 10
levels = 2
levels_actual = 2

phi = (1 + math.sqrt(5)) / 2
d_max_pct = 1 / (phi ** 2)
database = "/Users/jirka/Documents/forex_v1.db"
date_from = "\'2023-05-08\'"
date_to = "\'2023-05-25\'"
manual_dm = True
plot_optimize = True
plot_heatmap = True
plot_dm = True
plot_sum = True
# filter for optim.
my_code2 = """
def optim_func(series):
    if series[{}] <= {}: 
        return -1
    else:
        return series[{}]
      """.format("'# Trades'", min_trades, "'Win Rate [%]'")  # ,"'# Trades'" )

# if (na_filter == True): filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0) & (df_tp['na'] == 0)]

my_code = """
def optim_func(series):
        return series[{}]*series[{}]/100
      """.format("'Win Rate [%]'", "'# Trades'")  # ,"'# Trades'" )

exec(my_code)

"""
STRING_LIST1 = ["EURUSD","USDCHF","GBPUSD","AUDUSD"]
STRING_LIST2 = ["XAUUSD","USDJPY","SPXUSD","BCOUSD"]

for s in STRING_LIST1:
    if s == item_name:
        digits = 5
        d_min = 20
        d_max = 150
        min_size = 20  # pips
        csv_name = item_name
        table_name = item_name
for s in STRING_LIST2:
    if s == item_name:
        digits = 3
        d_min = 90
        d_max = 200
        min_size = 90  # pips
        csv_name = item_name
        table_name = item_name
"""
"""      
if item_name == "EURUSD" or "USDCHF" or "GBPUSD" or "AUDUSD":
    digits = 5
    d_min = 20
    d_max = 150
    min_size = 20 #pips
    csv_name = item_name
    table_name = item_name
elif item_name == "XAUUSD" or "USDJPY" or "SPXUSD" or "BCOUSD":
    digits = 3
    d_min = 500
    d_max = 1000
    min_size = 20 #pips
    csv_name = item_name
    table_name = item_name
"""

logger.info("Item name : {}", item_name)


class signal:
    def __init__(self, type, sl, enter, exit):
        self.type = type
        self.sl = sl
        self.enter = enter
        self.exit = exit


signals = []


class Item:
    def __init__(self, name, digits, d_min, d_max, min_size):
        self.name = name
        self.digits = digits
        self.d_min = d_min
        self.d_max = d_max
        self.min_size = min_size


try:
    del myList
except:
    pass

myList = []
#                item digits d_min d_max min_size
myList.append(Item("EURUSD", 5, 20, 80, 20))  # fibo čísla
myList.append(Item("USDCHF", 5, 36, 54, 36))
myList.append(Item("AUDUSD", 5, 36, 54, 36))
myList.append(Item("GBPUSD", 5, 36, 54, 36))

myList.append(Item("USDJPY", 3, 50, 90, 50))
myList.append(Item("XAUUSD", 3, 800, 1300, 800))
myList.append(Item("SPXUSD", 3, 1000, 1500, 1000))
myList.append(Item("BCOUSD", 3, 100, 300, 000))


def search_list_by_name(name, my_list):
    for obj in my_list:
        if obj.name == name:
            return obj
    return None


def get_last_period(df, period):
    last_index = pd.to_datetime(df.index[-1])
    last_period = df[(df.index >= pd.Timestamp(date(last_index.year, last_index.month, last_index.day)))]
    last_last = pd.to_datetime(last_period.index[-1])
    last_first = pd.to_datetime(last_period.index[0])
    return last_period, last_last, last_first


def get_first_period(df, period):
    # prev day
    first_index = pd.to_datetime(df.index[0])
    first_ = datetime(first_index.year, first_index.month, first_index.day, hour=0, minute=0, second=0)
    last_ = datetime(first_index.year, first_index.month, first_index.day, hour=23, minute=59, second=59)
    first_period = df[(df.index >= pd.Timestamp(first_)) & (df.index <= pd.Timestamp(last_))]
    first_period = first_period.sort_index()
    prev_period = first_period
    df.drop(df.head(len(first_period)).index, inplace=True)  # delete last period
    # last day
    first_index = pd.to_datetime(df.index[0])
    first_ = datetime(first_index.year, first_index.month, first_index.day, hour=0, minute=0, second=0)
    last_ = datetime(first_index.year, first_index.month, first_index.day, hour=23, minute=59, second=59)
    first_period = df[(df.index >= pd.Timestamp(first_)) & (df.index <= pd.Timestamp(last_))]
    first_period = first_period.sort_index()
    last_period = first_period
    return prev_period, last_period


def slev_levels_single(level, d, mean):
    # s = slev_levels(level = 1,d = 1.31417 - 1.29713, mean = 1.30600, df=df)
    # return levels
    phi = (1 + math.sqrt(5)) / 2
    s = []
    count = level * 3 + 1
    pattern = np.r_[2, 1, 2]
    # vytvoření jednotkového vektoru SLEV
    s = np.append(s, phi ** (level + 1))
    for i in range(-level, level + 1):
        s = np.append(s, phi ** (pattern + abs(i)))
    dd = np.cumsum(s / phi * d)
    a = dd - dd[count] - ((dd[count + 1] - dd[count]) / 2) + mean
    return a


def get_last_OHLC_data_xAPI(item_name, sample_size, time_frame, local_time_zone):
    """
    get index OHLC history based on size, timeframe and convert to local time zone
    :param item_name: index eg."EURUSD"
    :param sample_size: sample size per time frame eg. minutes 43200 = month M1, 2880 minutes = 2 days
    :param time_frame: time frame in seconds eg. 60 = minute
    :param local_time_zone: Prague
    :return: dataframe
    """
    logger.info("Item {} Size {} Frame {}", item_name, sample_size, time_frame)
    price_hl = client.get_lastn_candle_history(item_name, time_frame, sample_size)  # seconds interval, number records
    lst = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    # Calling DataFrame constructor on list
    df = pd.DataFrame(columns=lst)
    # get local time
    now_dt = datetime.now()
    utc_now = datetime.utcnow()
    now_ts, utc_ts = map(time.mktime, map(datetime.timetuple, (now_dt, utc_now)))
    offset = int((now_ts - utc_ts) / 3600)
    for i in price_hl:
        # time = datetime.datetime.utcfromtimestamp(i["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        local_time = datetime.fromtimestamp(i["timestamp"], timezone.utc) + timedelta(hours=offset)
        local_time = local_time.strftime('%Y-%m-%d %H:%M:%S')  #
        # open, close, high, low,, volume
        new_row = {'Datetime': local_time, 'Open': i["open"], 'High': i["high"], 'Low': i["low"],
                   'Close': i["close"],
                   'Volume': i["volume"]}
        df = df._append(new_row, ignore_index=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index("Datetime", inplace=True)
    return df


def any_data(data):
    return data


def get_last_2_days_XTB_data(df):
    logger.info("Login XTB...")
    # -------- GET LAST 2 DAYS DATA
    # day 0
    last_date = df.index[-1]
    last_date_b = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
    last_date_e = last_date.replace(hour=23, minute=59, second=59, microsecond=0)
    ts_today = df.loc[str(last_date_b):str(last_date_e)]
    # day -1
    prev_date = df.index[-len(ts_today) - 1]
    prev_date_b = prev_date.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_date_e = prev_date.replace(hour=23, minute=59, second=59, microsecond=0)
    ts_prev = df.loc[str(prev_date_b):str(prev_date_e)]
    return ts_today, ts_prev
    # bcktesting
    # from table item - params
    # range d,m
    # CALCULATE THE PARAMETERS D-RANGE, M-RANGE


def get_dm_range(ts):
    max_price = int(math.pow(10, obj.digits) * ts["High"].max())
    min_price = int(math.pow(10, obj.digits) * ts["Low"].min())
    mean_price = int(math.pow(10, obj.digits) * ts["Close"].median())

    HL_distance = int((max_price - min_price))

    max_price = int(max_price + HL_distance)
    min_price = int(min_price - HL_distance)

    c_range = int((max_price - min_price) / scale)
    d_max = int((max_price - min_price) * d_max_pct)
    d_size = int((d_max - obj.min_size) / scale)

    d_range = range(obj.min_size, d_max, d_size)
    m_range = range(min_price, max_price, c_range)
    # m_range = range(mean_price - 200, mean_price + 200,1)

    logger.info("DM range D={} M={}", d_range, m_range)
    return d_range, m_range


def plot_heatmap_all(name):
    output_file(name)
    aa = heatmap.to_frame()
    aa = aa.reset_index()
    open = prev_period.iloc[0]["Open"]
    close = prev_period.iloc[-1]["Close"]
    c = np.where(aa[0].values < 0, "#FFFFFF", "#000000")
    p = figure()
    p.scatter(x=aa["d"], y=aa["m"] / 10 ** obj.digits, color=c)
    p.hspan(y=open, line_width=3, line_color="blue")
    p.hspan(y=close, line_width=3, line_color="red")
    p.vspan(x=d, line_width=3, line_color="black")
    p.hspan(y=m / 10 ** obj.digits, line_width=3, line_color="black")
    show(p)
    return


# ---------- OPTIMIZE RANGE(D,M)
def Backtest_optimize(ts, level, dm, mm, plot):
    class SignalStrategy(Strategy):
        d = 0
        m = 0

        def init(self):
            signals.clear()  # nutno čistit jinak se vedme globální hodnota
            a = slev_levels_single(level, self.d / 10 ** obj.digits, self.m / 10 ** obj.digits)
            a = np.round(a, obj.digits)
            for i in a:
                self.bid = self.I(any_data, np.repeat(i, len(self.data)), name=str(i), color='black')
            l = len(a)
            x = range(0, l - 3, 3)
            for n in x:
                # sl, enter, exit
                signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
                signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))

        def next(self):
            if not self.position:
                for obj in signals:
                    if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "BUY":
                        self.buy(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen up směr
                        break
                    if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "SELL":
                        self.sell(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen down smšr
                        break

    """
    def optim_func(series):
        if series['# Trades'] <= min_trades: #min lost orders, max profit zvýši P že není náhoda
            return -1
        else:
            return series['# Trades']*series['Win Rate [%]']

    """
    bt = Backtest(ts, SignalStrategy)
    stat, heatmap = bt.optimize(d=dm, m=mm, maximize=optim_func,
                                return_heatmap=True)  # pozor na počet digits, musí odpovídat pak /100
    if plot:
        filename = str(ts.index[-1])
        filename = filename[:-9]
        bt.plot(filename="O" + filename, plot_pl=True, show_legend=False, superimpose=False)
        if plot_heatmap:
            plot_heatmaps(heatmap, filename="H" + filename + ".html")
        # hm = heatmap.groupby(["d", "m"]).mean().unstack()
        # sns.heatmap(hm.T, cmap="plasma")
        # plt.show()
        logger.info("Previous Day Backtest Optimize plot ")
        # the best trade
        logger.info(" ---- Win strategy -------")
        logger.info("Optimal D: {}", stat._strategy.d)
        logger.info("Optimal M: {}", stat._strategy.m)
        logger.info("Optimal orders: \n {}", stat.to_string())
        a = slev_levels_single(levels, stat._strategy.d / 100000, stat._strategy.m / 100000)
        logger.info("A-levels: {}", a)
        logger.info("Last Day Prediction plot ")
    return stat._strategy.d, stat._strategy.m, stat._trades, heatmap


# ---------- Backtest D,M
def Backtest_dm(ts, level, d_, m_, plot):
    class SignalStrategy_opt(Strategy):
        d = d_
        m = m_

        def init(self):
            signals.clear()  # nutno čistit jinak se vedme globální hodnota
            a = slev_levels_single(level, self.d / 10 ** obj.digits, self.m / 10 ** obj.digits)
            a = np.round(a, obj.digits)
            for i in a:
                self.bid = self.I(any_data, np.repeat(i, len(self.data)), name=str(i), color='black')
            l = len(a)
            x = range(0, l - 3, 3)
            for n in x:
                # sl, enter, exit
                signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
                signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))

        def next(self):
            if not self.position:
                for obj in signals:
                    if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "BUY":
                        self.buy(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen up směr
                        break
                    if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "SELL":
                        self.sell(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen down smšr
                        break

    bt_org = Backtest(ts, SignalStrategy_opt)
    output = bt_org.run()
    if plot:
        logger.info("Last Day Stats {}", output)
        logger.info("Last Day Heatmap plot ")
        filename = "L" + str(ts.index[-1])
        filename = filename[:-9]
        bt_org.plot(show_legend=False, filename=filename, superimpose=False)
        pd.set_option('display.max_columns', None)
        # print(output._trades)
    return output._trades, output


if __name__ == "__main__":
    # get SQLite data
    # READ FILE DAT_MT_xxxxx
    obj = search_list_by_name(item_name, myList)
    sql_command = f"SELECT * FROM  {obj.name} WHERE Datetime BETWEEN {date_from} AND {date_to}"
    conn = sqlite3.connect(database)
    df_ = pd.read_sql_query(sql_command, conn)
    conn.close()
    logger.info("\n SQL : {} \n lenth : {}", sql_command, len(df_))
    df_['Datetime'] = pd.to_datetime(df_['Datetime'])
    df_.set_index("Datetime", inplace=True)
    df_ = df_.sort_index()
    df_copy = df_.copy()

    # trades dataframe
    # adjust level based on H,L(max,mix) vs a(min max)

    # LOOP
    column_names = ['Size', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'EntryTime',
                    'ExitTime', 'Duration']
    df_trades = pd.DataFrame(columns=column_names)

    while True:
        try:
            prev_period, last_period = get_first_period(df_, time_frame)
            logger.info("\n prev period: {} \n last period: {}", prev_period.index[-1], last_period.index[-1])
            # check if prev not > last !!!!
            dm, mm = get_dm_range(prev_period)
            if manual_dm == True:
                dm = range(obj.d_min, obj.d_max, 1)
            logger.info("D range: {}  M range: {}", dm, mm)
            """
            #reverse column data
            prev_period["Open"] = prev_period["Open"].values[::-1]
            prev_period["High"] = prev_period["High"].values[::-1]
            prev_period["Low"] = prev_period["Low"].values[::-1]
            prev_period["Close"] = prev_period["Close"].values[::-1]
            """
            d, m, _trades, heatmap = Backtest_optimize(prev_period, levels, dm, mm, plot_optimize)
            """
            mm = range(105042, 105948, 1)
            dm = range(20,30)
            d, m, _trades, heatmap = Backtest_optimize(prev_period, levels, dm, mm, plot_optimize)
            """
            if heatmap.max() <= 0:
                logger.info("Low 'Return [%]': {} skip iteration", heatmap.max())
                continue
            # save heatmap
            filename = "H_" + item_name + "_" + str(prev_period.index[-1])
            filename = filename[:-9] + "_" + str(scale)
            np.save(filename + '.npy', heatmap)
            # heatmap.to_hdf(filename, key='df', mode='w')
            # heatmap_load = np.load(filename)
            logger.info("Optimized d : {}  m : {}", d, m)
            # backtest data
            # m = prev_period["Close"].iloc[-1] * (10 ** digits)   když do CLOSE!!
            trades, output = Backtest_dm(last_period, 4, d, m, plot_dm)
            trades["D"] = round(d / 10 ** obj.digits, obj.digits)
            trades["M"] = round(m / 10 ** obj.digits, obj.digits)
            df_trades = df_trades._append(trades)
            logger.info("Add orders : {}", len(trades))
        except Exception as error:
            logger.error("End of histdata {} - {} ", type(error).__name__, "–", error, error)
            break

    # plot_heatmap_all("HA" + filename + ".html")

    # ---------------- FINAL RESULTS - STATS
    logger.info("Plot final results...")
    _succesful = len(df_trades[(df_trades["PnL"] > 0)])
    _all_orders = len(df_trades)
    """
    PLOT ALL TRADES AND STATS
    """

    df_trades = df_trades.reset_index()
    df_copy["ExitTime"] = 0
    df_copy["EntryTime"] = 0
    df_copy["EntryPrice"] = 0
    df_copy["ExitPrice"] = 0
    df_copy["Signal"] = 0
    df_copy["Return"] = 0
    df_copy["ReturnPct"] = 0
    for i in range(0, len(df_trades)):
        # kdyz return < 0 exit je SL
        df_copy["EntryPrice"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                         df_trades["EntryPrice"].iloc[i], df_copy['EntryPrice'])
        df_copy["ExitPrice"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                        df_trades["ExitPrice"].iloc[i], df_copy['ExitPrice'])
        df_copy["Signal"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                     np.where(df_trades["EntryPrice"].iloc[i] < df_trades["ExitPrice"].iloc[i], 1, -1),
                                     df_copy['Signal'])
        df_copy["Return"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                     np.where(df_trades["PnL"].iloc[i] > 0,
                                              abs(df_trades["ExitPrice"].iloc[i] - df_trades["EntryPrice"].iloc[i]),
                                              -abs(df_trades["ExitPrice"].iloc[i] - df_trades["EntryPrice"].iloc[i])),
                                     df_copy['Return'])
        df_copy["ExitTime"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                       df_trades["ExitTime"].iloc[i], df_copy['ExitTime'])
        df_copy["EntryTime"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                        df_trades["EntryTime"].iloc[i], df_copy['EntryTime'])
        df_copy["ReturnPct"] = np.where((df_copy.index == df_trades["EntryTime"].iloc[i]),
                                        df_trades["ReturnPct"].iloc[i], df_copy['ReturnPct'])
    df_copy = df_copy.sort_index()


    class SignalStrategy_final(Strategy):
        def init(self):
            pass

        def next(self):
            current_signal = self.data.Signal[-1]  # 1 BUY -1 SELL
            return_signal = self.data.Return[-1]  # 1 tp = exit_price Profit -1 Loss / sl = exit price
            if return_signal > 0:
                if current_signal == 1:
                    self.buy(tp=self.data.ExitPrice[-1])
                elif current_signal == -1:
                    self.sell(tp=self.data.ExitPrice[-1])
            if return_signal < 0:
                if current_signal == 1:
                    self.sell(sl=self.data.ExitPrice[-1])
                elif current_signal == -1:
                    self.buy(sl=self.data.ExitPrice[-1])


    bt = Backtest(df_copy, SignalStrategy_final)
    stats = bt.run()
    if plot_sum: bt.plot()

    filename = "F" + str(scale) + item_name + str(df_copy.index[-1])
    filename = filename[:-9] + ".html"

    output_file(filename)

    # anotations
    item_params = f"Item: {obj.name} /  digits: {obj.digits} scale: {scale}  \n" \
                  f"Interval: {date_from} - {date_to}  \n" \
                  f"Levels forecast: {levels} last: {levels_actual} \n" \
                  f"d.interval {obj.d_min} - {obj.d_max} success ord. {_succesful} from {_all_orders} \n" \
                  f"database:  {database} \n" \
                  f"filter: \n {my_code} \n\n"

    output_text = item_params + str(stats)
    statistics = PreText(text=output_text, width=600, height=100)
    # Return
    s1 = figure(title='Cumul. Return', x_axis_label='DateTime', y_axis_label='PnL', x_axis_type='datetime', width=400,
                height=400)
    # add a circle renderer with a size, color, and alpha
    s1.circle(df_trades["ExitTime"], df_trades["PnL"].cumsum(), size=2, color="navy", alpha=0.5)

    # source = ColumnDataSource(df_trades)
    # del df_trades["level_0"]
    # p.line('ExitTime', 'PnL', source=source)
    # show the results
    # s1.add_layout(Label(x=df_trades["ExitTime"][0], y=df_trades["PnL"].cumsum().iloc[-1], text=sql_command, text_font_size="8px"))  ## Label 1
    # p.add_layout(mytext)

    # Signals

    # source = ColumnDataSource(df_trades)
    # del df_trades["level_0"]
    # p.line('ExitTime', 'PnL', source=source)
    # show the results
    # p.add_layout(mytext)

    # Signals
    s3 = figure(title='D', x_axis_label='DateTime', y_axis_label='Signal', x_axis_type='datetime',
                width=400, height=400)
    s3.line(df_trades["ExitTime"].sort_values(), df_trades["D"].values * 10000)
    # s3.line(df_trades["ExitTime"].sort_values(), df_trades["M"])

    # Signals
    s4 = figure(title='M', x_axis_label='DateTime', y_axis_label='Signal', x_axis_type='datetime',
                width=400, height=400)
    s4.line(df_trades["ExitTime"].sort_values(), df_trades["M"].values)
    # s3.line(df_trades["ExitTime"].sort_values(), df_trades["M"])

    show(row(statistics, column(s1, s3, s4)))

    filename = "F" + str(scale) + item_name + str(df_copy.index[-1])
    filename = filename[:-9] + ".csv"
    df_trades.to_csv(filename)

# plt.plot(df_copy["Return"].cumsum())
# plt.show()

output_file("hourly.html")
"""

#store Pandas Sdries
heatmap.to_hdf('data.h5', key='df', mode='w')
heatmap_load = pd.read_hdf('H_EURUSD_2023-10-24_50', 'df')
heatmap_load.idxmax()
heatmap_load.nlargest()
heatmap_load.sort_values(ascending=False)
heatmap_load.pct_change(ascending=False)

plot = heatmap_load.plot(kind="kde")
plt.show()

heatmap_load.hist()
plt.show()

plt.scatter(heatmap_load[0],heatmap_load[:,1],heatmap_load[:,2], edgecolor='red', s=40, alpha = 0.5)
heatmap_load.index[0][0] #d
heatmap_load.index[0][1] #m
heatmap_load.values[4] #value

heatmap_load.index.values
heatmap_load.index.get_level_values(0)
heatmap_load.index.get_level_values(1)
heatmap_load.values
plt.scatter(heatmap_load.index.get_level_values(0), heatmap_load.index.get_level_values(1), col = heatmap_load.values)
plt.show()


ax = plt.figure().add_subplot(projection='3d')

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(X, Y, Z, colormap='viridis')
ax.set_title('3D Scatter Plot')

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('t', labelpad=20)

plt.show()

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)

X = heatmap_load.index.get_level_values(0)
Y = heatmap_load.index.get_level_values(1)
Z = heatmap_load.values

my_cmap = plt.get_cmap('viridis')

# Creating plot
sctt = ax.scatter3D(X, Y, Z, cmap = my_cmap, c = Z)
plt.title("simple 3D scatter plot")
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
# show plot
plt.show()

"""
fig = figure(x_axis_type='datetime')
c = np.where((df_copy["Return"] < 0), "#FF0000", df_copy["Return"])
c = np.where((df_copy["Return"] > 0), "#0000FF", c)
# d = np.where(df_copy["Signal"] == 1, "#0000FF", 0)
fig.scatter(df_copy.index, df_copy["Close"], size=1)
fig.scatter(df_copy.index, df_copy["Close"], color=c, size=np.pi * (10 * df_copy["Return"] * 100) ** 2)

# only hours filter
df2 = df_copy.copy()
df2.index = pd.to_datetime(df2.index).strftime('%H:%M:%S')  # %Y-%m-%d %H:%M:%S
df2 = df2.set_index(pd.DatetimeIndex(df2.index))
df2 = df2.sort_index()

fig2 = figure(x_axis_type='datetime')
c = np.where((df2["Return"] < 0), "#FF0000", df2["Return"])
c = np.where((df2["Return"] > 0), "#0000FF", c)
# d = np.where(df_copy["Signal"] == 1, "#0000FF", 0)
fig2.scatter(df2.index, df2["Close"], color=c, size=np.pi * (10 * df2["Return"] * 100) ** 2)

daily_stat = df2["Return"].groupby(pd.Grouper(freq='1H')).sum()
c = np.where(daily_stat.values < 0, "#FF0000", "#0000FF")
fig3 = figure(x_axis_type='datetime', height=250)
fig3.vbar(x=daily_stat.index, top=daily_stat.values * 100, width=10, color=c)
fig3.xgrid.grid_line_color = None
fig3.y_range.start = -1

show(row(fig, column(fig2, fig3)))

"""
from itertools import combinations
param_combinations = combinations(heatmap.index.names, 2)
dfs = [heatmap.groupby(list(dims)).agg("max").to_frame(name='_Value')
       for dims in param_combinations]
plots = []
cmap = LinearColorMapper(palette='Viridis256',
                         low=min(df.min().min() for df in dfs),
                         high=max(df.max().max() for df in dfs),
                         nan_color='white')
for df in dfs:
    name1, name2 = df.index.names
    level1 = df.index.levels[0].astype(str).tolist()
    level2 = df.index.levels[1].astype(str).tolist()
    df = df.reset_index()
    df[name1] = df[name1].astype('str')
    df[name2] = df[name2].astype('str')

    fig = figure(x_range=level1,
                  y_range=level2,
                  x_axis_label=name1,
                  y_axis_label=name2,
                  width=plot_width // ncols,
                  height=plot_width // ncols,
                  tools='box_zoom,reset,save',
                  tooltips=[(name1, '@' + name1),
                            (name2, '@' + name2),
                            ('Value', '@_Value{0.[000]}')])
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_standoff = 0

fig.rect(x=name1,
                 y=name2,
                 width=1,
                 height=1,
                 source=df,
                 line_color=None,
                 fill_color=dict(field='_Value',
                                 transform=cmap))
show(fig, browser=None if open_browser else 'none')
return fig
"""





