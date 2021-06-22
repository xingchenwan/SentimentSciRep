# Xingchen Wan | Oxford-Man Institute of Quantitative Finance | Updated 6 Sep 2018

import processor, grouper, analyser, visualiser, utilities
from source.all_classes import *
import matplotlib.pylab as plt
import pandas as pd

pd.options.display.max_rows = 10000

NAME = 'GS'
MODE = 'rtn'
# Sector representatives:
# Financials: AXP | Largest: GS
# Tech: AAPL
# Retail: WMT
# Industrials: BA | Largest: GE


# Cluster representative:
# 0. GS
# 1. GM
# 2. GE
# 3. HD
# 4. BA
# 5. XOM
# 6. AA

names = ['GS', 'GM', 'GE', 'HD', 'BA', 'XOM', 'AA']

# path to the full_data object. see API of this in source/all_classes.py
full_data_obj = 'data/full.date.20061020-20131120.mergeJPM'
# market data file path. A Microsoft Excel file expected.
market_data_path = 'data/market_data.xlsx'
# the tab name with the market data
market_data_sheet = 'Price1'

# sector path csv
sector_path = "source/sector.csv"

START_DATE = datetime.date(2007, 1, 1)
END_DATE = datetime.date(2013, 12, 31)
MAX_LAG = 7

PATH_POS = "output/Sep2019/" + "posEvents" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
           str(START_DATE) + "_" + str(END_DATE) + ".csv"
PATH_NEG = "output/Sep2019/" + "negEvents" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
           str(START_DATE) + "_" + str(END_DATE) + ".csv"
FIGURE_PATH = "figures/" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
              str(START_DATE) + "_" + str(END_DATE) + ".png"


def event(start_date=START_DATE, end_date=END_DATE):
    pd.options.display.max_rows = 10000
    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    sub_all_data = utilities.create_sub_obj(all_data, start_date, end_date)
    price, daily_rtn, log_return, vol, _, _ = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                                                   start_date=pd.to_datetime(
                                                                                       start_date),
                                                                                   end_date=pd.to_datetime(end_date),
                                                                                   )
    all_names = list(price.columns)[:-1]
    sector = grouper.get_grouping_from_csv("dynamic_grouping2007-01-012007-12-31.csv")

    for name in names:
        # Static Grouping
        this_sector = grouper.get_sector_peer(sector, name)
        # Dynamic Grouping

        (count, med_sentiment, sum_sentiment,) = processor.process_count_sentiment(sub_all_data,
                                                                                   start_date=pd.to_datetime(start_date),
                                                                                   end_date=pd.to_datetime(end_date),
                                                                                   focus_iterable=this_sector,
                                                                                   rolling=True,
                                                                                   rolling_smoothing_factor=0.7)

        if MODE == 'vol':
            pos, neg, _, _ = analyser.event_analyser(this_sector, market_frame=vol, exogen_frame=sum_sentiment, mode=MODE,
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     max_lag=MAX_LAG,
                                                     file_name=name)
        elif MODE == 'rtn':
            pos, neg, _, _ = analyser.event_analyser(this_sector, market_frame=daily_rtn, exogen_frame=sum_sentiment,
                                                     mode=MODE,
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     max_lag=MAX_LAG,
                                                     file_name=name)
        else:
            raise ValueError()


def event_sentiment_nearest_neighbour_diffusion(k_nn=10, unpickle: str = None):
    """Similar to the def that computes the sentiment diffusion from an entity to its sector peers, this def insteads
    computes the corresponding quantity from an entity to nearest neighbours

    k_nn: the number (k) of the top nearest neighbours to return
    """
    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    if START_DATE is not None or END_DATE is not None:
        sub_all_data = utilities.create_sub_obj(all_data, START_DATE, END_DATE)
    else:
        sub_all_data = all_data  #
    price, daily_rtn, log_return, vol, _, _ = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                                                   start_date=pd.to_datetime(
                                                                                       START_DATE),
                                                                                   end_date=pd.to_datetime(END_DATE), )
    all_names = list(price.columns)[:-1]
    sector = grouper.get_sector_grouping(all_names, sector_path, 0, 2)
    this_sector = grouper.get_sector_peer(sector, NAME)
    if not unpickle:
        nearest_neighbours = grouper.get_nearest_neighbour(this_sector, sub_all_data, all_names, START_DATE, END_DATE, top_n=k_nn)
    else:
        nearest_neighbours = pickle.load(open(unpickle, 'rb'))
    (count, med_sentiment, sum_sentiment,) = processor.process_count_sentiment(sub_all_data,
                                                                               start_date=pd.to_datetime(START_DATE),
                                                                               end_date=pd.to_datetime(END_DATE),
                                                                               focus_iterable=all_names,
                                                                               rolling=True,
                                                                               rolling_smoothing_factor=0.7)
    analyser.nearest_neighbour_sentiment_analyser(this_sector, nearest_neighbours, sum_sentiment, start_date=START_DATE,
                                                  end_date=END_DATE, file_name=NAME, )


def event_sentiment_diffusion(start_date=START_DATE, end_date=END_DATE):
    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    if start_date is not None or end_date is not None:
        sub_all_data = utilities.create_sub_obj(all_data, start_date, end_date)
    else:
        sub_all_data = all_data  #

    # use the market data frame just to get the column names - super ugly and inefficient but this will be
    # ditched later...
    price, daily_rtn, log_return, vol, _, _ = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                                                   start_date=pd.to_datetime(
                                                                                       start_date),
                                                                                   end_date=pd.to_datetime(end_date), )
    all_names = list(price.columns)[:-1]

    sector = grouper.get_sector_grouping(all_names, sector_path, 0, 2)
    # sector = grouper.get_grouping_from_csv("dynamic_grouping2007-01-012013-11-20.csv")
    sector = grouper.get_sector_grouping(all_names, sector_path, 0, 2)
    this_sector = grouper.get_sector_peer(sector, NAME)

    (count, med_sentiment, sum_sentiment,) = processor.process_count_sentiment(sub_all_data,
                                                                               start_date=pd.to_datetime(start_date),
                                                                               end_date=pd.to_datetime(end_date),
                                                                               focus_iterable=this_sector,
                                                                               rolling=True,
                                                                               rolling_smoothing_factor=0.7)

    analyser.group_sentiment_analyser(this_sector, med_sentiment, start_date=START_DATE, end_date=END_DATE,
                                      save_csv=True, file_name=NAME, max_lag=7)


def event_px_diffusion(start_date=START_DATE, end_date=END_DATE):
    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    if start_date is not None or end_date is not None:
        sub_all_data = utilities.create_sub_obj(all_data, start_date, end_date)
    else:
        sub_all_data = all_data  #

    # use the market data frame just to get the column names - super ugly and inefficient but this will be
    # ditched later...
    price, daily_rtn, log_return, vol, volume, market_cap = processor.process_market_time_series \
        (market_data_path, market_data_sheet, start_date=pd.to_datetime(start_date), end_date=pd.to_datetime(end_date),)

    all_names = list(price.columns)[:-1]
    # sector = grouper.get_grouping_from_csv("dynamic_grouping2007-01-012013-11-20.csv")
    sector = grouper.get_grouping_from_csv('dynamic_grouping2007-01-012007-12-31.csv')
    for name in names:
        this_sector = grouper.get_sector_peer(sector, name)
        # sector = grouper.get_grouping_from_csv("dynamic_grouping2007-01-012013-11-20.csv")

        # Static Grouping
        # this_sector = grouper.get_sector_peer(sector, NAME)

        count, med_sentiment, sum_sentiment = processor.process_count_sentiment(sub_all_data,
                                                                                start_date=pd.to_datetime(start_date),
                                                                                end_date=pd.to_datetime(end_date),
                                                                                focus_iterable=this_sector,
                                                                                rolling=True,
                                                                                rolling_smoothing_factor=0.7)
        if MODE == 'vol':
            analyser.group_average(this_sector, vol, sum_sentiment, start_date=START_DATE, end_date=END_DATE,
                                   save_csv=True, file_name=name, max_lag=7, mode=MODE, volume_frame=market_cap,
                                   market_cap_frame=market_cap, exclude_large_cap=0,
                                   aggregate_price=False)
        elif MODE == 'rtn':
            analyser.group_average(this_sector, daily_rtn, sum_sentiment, start_date=START_DATE, end_date=END_DATE,
                                   save_csv=True, file_name=name, max_lag=7, mode=MODE, volume_frame=market_cap,
                                   market_cap_frame=market_cap, exclude_large_cap=0,
                                   aggregate_price=False)


def event_by_year(start_date):
    for _ in range(7):
        end_date = start_date + datetime.timedelta(days=365)
        print(start_date, end_date)
        event(start_date, end_date)
        start_date += datetime.timedelta(days=365)


def plot_sentiment_time_series(name):
    START_DATE = datetime.date(2011, 4, 15)
    END_DATE = datetime.date(2011, 5, 15)
    full_data_obj = 'data/full.date.20061020-20131120'
    pd.options.display.max_rows = 10000

    fig, ax = plt.subplots()
    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)
    if START_DATE is not None or END_DATE is not None:
        sub_all_data = utilities.create_sub_obj(all_data, START_DATE, END_DATE)
    else:
        sub_all_data = all_data  #
    (count, med_sentiment, sum_sentiment,) = processor.process_count_sentiment(sub_all_data,
                                                                               focus_iterable=[name],
                                                                               rolling=True,
                                                                               rolling_smoothing_factor=0.7)
    plt.plot(sum_sentiment)
    fig.autofmt_xdate()
