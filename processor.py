# Processor module: process relevant market and sentiment data for subsequent analyses
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import pandas as pd
import numpy as np
from collections import Counter
import datetime


def process_beta_time_series(path, start_date=None, end_date=None, period="weekly"):
    df = pd.read_csv(path)
    assert all(string in df.columns for string in ['Date', 'Close'])
    df = df[['Date', 'Close']]
    # Initialise an datetime index
    df_start_date = df['Date'].iloc[0]
    df_end_date = df['Date'].iloc[-1]

    # Change the datetime string to pandas datetime object
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.set_index('Date')

    # Set the start and end dates. The start date the later date between the "start_date" argument and the dataframe
    # start date. The end date is the earlier date between the "end_Date" argument and the dataframe end date
    start_date = max(df_start_date, start_date) if start_date is not None else df_start_date
    end_date = min(df_end_date, end_date) if end_date is not None else df_end_date

    # Create the pandas date index
    date_idx = pd.date_range(start_date, end_date, freq='D')

    # Initialise the series to return
    data = df.reindex(date_idx)
    data = data.fillna(method='ffill') # Forward fill the data entries

    # Compute log return, period vol, daily_vol and etc
    daily_log_return = data.apply(lambda x: np.log(x.shift(-1)) - np.log(x))
    if period == 'weekly':
        period_log_return = data.apply(lambda x: np.log(x.shift(-5)) - np.log(x))
    else:
        raise TypeError()
    vol = daily_log_return.abs()
    return data, daily_log_return, period_log_return, vol


def process_market_time_series(path, px_sheet=None, volume_sheet=None, market_cap_sheet=None,
                               start_date=None, end_date=None, period="weekly",
                               focus_iterable=None):
    """
    Process the raw price series from Bloomberg
    :param path: path string of the price series document
    :param px_sheet: Sheet name in the excel document
    :param start_date: datetime.date. Start date of the processed series
    :param end_date: datetime.date. End date of the processed series
    :param period: Valid str literal or int. Period to calculate log return. Default is "7D". Recommend to use 7D
    rather than 7 due to the difference in convention between calendar days and trading days.
    :param market_cap_sheet: Sheet name of the market cap tab in the excel document
    :param volume_sheet: sheet name of the volume tab in the excel document
    :param focus_iterable: the list of securities and only market data related to these securities are returned and other
    market data are discarded. By default: None - all market data will be returned

    :return:
    data: Array-like. the unprocessed price time series
    daily_log_return: Array-like. the daily log return series
    period_log_return: Array-like. the log return series in accordance to the set period. Default is weekly log return
    vol: Array-like. volatility of daily log return in period time. Default is weekly realised volatility
    volume, market_cap: array-like. The volume and market_cap time series pruned based on focus iterables and start and
    end dates specified
    """
    import xlrd

    def read_date(date):
        """
        In case conversion from the Excel-style datetime is needed - not currently in use
        :param date: Excel datetime object
        :return: Pandas-compliant datetime data type
        """
        return xlrd.xldate.xldate_as_datetime(date, 0)
    if period == 'weekly':
        roll_period = '7D'
        shift_period = 5  # Trading week
    elif period == 'daily':
        roll_period = '1D'
        shift_period = 1
    elif period == '2d':
        roll_period = '2d'
        shift_period = 2
    elif period == 'monthly':
        roll_period = '30D'
        shift_period = 21  # Trading month
    else:
        roll_period = str(period) + 'D'
        shift_period = int(period / 7 * 5)

    raw = pd.read_excel(path, sheet_name=px_sheet, index_col=0)
    # print(raw)
    start_date = start_date if start_date and start_date >= raw.index[1] else raw.index[1]
    end_date = end_date if end_date and end_date <= raw.index[-1] else raw.index[-1]
    raw = raw[raw.index >= start_date]
    data = raw[raw.index <= end_date].astype('float64')
    if focus_iterable:
        focus_securities = [x for x in data.columns if x in focus_iterable]
        data = data[focus_securities]
    # Truncate the price time series to be consistent with the count and sentiment time series
    daily_log_return = data.apply(lambda x: np.log(x.shift(-1)) - np.log(x))
    period_log_return = data.apply(lambda x: np.log(x.shift(-shift_period)) - np.log(x))
    vol = daily_log_return.abs()

    if volume_sheet is not None:
        volume = pd.read_excel(path, sheet_name=volume_sheet, index_col=0)
        start_date = start_date if start_date and start_date >= volume.index[1] else volume.index[1]
        end_date = end_date if end_date and end_date <= volume.index[-1] else volume.index[-1]
        volume = volume[volume.index >= start_date]
        volume = volume[volume.index <= end_date].astype('float64')
        if focus_iterable:
            focus_securities = [x for x in volume.columns if x in focus_iterable]
            volume = volume[focus_securities].abs()
    else:
        volume = None

    if market_cap_sheet is not None:
        market_cap = pd.read_excel(path, sheet_name=market_cap_sheet, index_col=0)
        start_date = start_date if start_date and start_date >= volume.index[1] else volume.index[1]
        end_date = end_date if end_date and end_date <= volume.index[-1] else volume.index[-1]
        market_cap = market_cap[market_cap.index >= start_date]
        market_cap = market_cap[market_cap.index <= end_date].astype('float64')
        if focus_iterable:
            focus_securities = [x for x in market_cap.columns if x in focus_iterable]
            market_cap = market_cap[focus_securities].abs()
    else:
        market_cap = None

    return data, daily_log_return, period_log_return, vol, volume, market_cap


def process_count_sentiment(full_data_obj, start_date=None, end_date=None, mode='weekly',
                            rolling=False, rolling_smoothing_factor=0.2, focus_iterable=None):
    """
    Process the FullData object to obtain corresponding count and sentiment time series
    :param full_data_obj: The FullData object containing all the unprocessed count and sentiment information.
    :param start_date: datetime, optional - start time of data. To be a valid argument, the start_date
    must be later than the start date of the FullData object
    :param end_date: datetime, optional - end time of the data. To be a valid argument, the end_date must be earlier
    than the end date of the FullData object
    :param mode: the period for aggregation. 'weekly','monthly', 'daily'
    :param focus_iterable: optional [] or {} - only entities in the list will be considered. If blank all entities
    will be considered
    :return:
    """

    def is_end_of_period(day, mode):
        from pandas.tseries.offsets import MonthEnd

        if mode == 'monthly': return True if day == day + MonthEnd(0) else False
        elif mode == 'daily': return True
        elif mode == 'weekly': return True if day.weekday() == 4 else False  # Snap to nearest Friday
        else: raise ValueError("Unrecognised mode. Only weekly, daily and monthly modes are currently supported.")

    def non_zero_median(list_obj):
        v = np.asarray(list_obj)
        #v = v[np.nonzero(v)]
        #return np.nanmedian(v) if len(v) else 0
        return np.nanmean(v)

    def moving_avg(list, alpha=rolling_smoothing_factor):
        series = pd.Series(list).fillna(0, inplace=True)
        ewm_series = series.ewm(alpha=alpha)
        return ewm_series.iloc[-1]

    # Misc sanity checks...
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date() if start_date.date() > full_data_obj.start_date else full_data_obj.start_date
    elif isinstance(start_date, datetime.date):
        start_date = start_date if start_date > full_data_obj.start_date else full_data_obj.start_date
    else:
        if start_date is not None:
            print('Invalid start date. Using the FullData object start date.')
        start_date = full_data_obj.start_date

    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date() if end_date.date() < full_data_obj.end_date else full_data_obj.end_date
    elif isinstance(end_date, datetime.date):
        end_date = end_date if end_date < full_data_obj.end_date else full_data_obj.end_date
    else:
        if end_date is not None:
            print('Invalid end date. Using the FullData object end date')
        end_date = full_data_obj.end_date

    if focus_iterable:
        if isinstance(focus_iterable, dict):
            focus_iterable = focus_iterable.keys()
        elif isinstance(focus_iterable, list):
            pass
        else:
            raise TypeError("include_list needs to be an iterable of type either list or dictionary")
    else:
        focus_iterable = full_data_obj.entity_occur_interval.keys()

    # Initialise the aggregate list and counter
    aggregate_count_sum = []
    aggregate_sentiment_med = []
    aggregate_sentiment_sum = []
    date_time_series = []

    # Initialise the weekly counters
    period_count_sum = Counter({})
    period_sentiment_sum = Counter({})
    period_sentiment_med = Counter({})

    # Setting appropriate period in terms of CALENDAR days
    if mode == 'daily': period = 1
    elif mode == 'weekly': period = 7
    elif mode == 'monthly': period = 30
    else: period = 7

    if rolling:
        # Rolling weekly counters
        for day in full_data_obj.days:
            if day.date >= start_date and day.date <= end_date:
                date_time_series.append(day.date)
                day_count_sum = Counter(dict((name, 0) for name in focus_iterable))
                day_sentiment_med = Counter(dict((name, 0) for name in focus_iterable))
                day_sentiment_sum = Counter(dict((name, 0) for name in focus_iterable))
                for name in focus_iterable:
                    if name in day.entity_occur_day.keys():
                        day_count_sum[name] = day.entity_occur_day[name]
                        day_sentiment_med[name] = day.entity_sentiment_day[name]
                        day_sentiment_sum[name] = day.entity_occur_day[name] * day.entity_sentiment_day[name]
                aggregate_count_sum.append(day_count_sum)
                aggregate_sentiment_med.append(day_sentiment_med)
                aggregate_sentiment_sum.append(day_sentiment_sum)

    else:
        # Non-rolling counters - weekly counter set at every Friday
        i = 0
        for day in full_data_obj.days:
            if day.date < start_date:
                continue
            elif day.date == start_date:
                i = 0
                period_count_sum = Counter(dict((name, 0) for name in focus_iterable))
                period_sentiment_med = Counter(dict((name, []) for name in focus_iterable))
                period_sentiment_sum = Counter(dict((name, 0) for name in focus_iterable))
            elif day.date > end_date:
                break

            if is_end_of_period(day.date, mode):
                aggregate_count_sum.append(period_count_sum)
                aggregate_sentiment_med.append({k: non_zero_median(v) if len(v) else 0 for k, v in period_sentiment_med.items()})
                aggregate_sentiment_sum.append(period_sentiment_sum)
                # Re-initialise the periodical sub-totals for the next period
                date_time_series.append(day.date)
                period_count_sum = Counter(dict((name, 0) for name in focus_iterable))
                period_sentiment_med = Counter(dict((name, []) for name in focus_iterable))
                period_sentiment_sum = Counter(dict((name, 0) for name in focus_iterable))

            else:
                for news in day.day_news:
                    for entity in focus_iterable:
                        if entity in news.entity_occur.keys():
                            period_count_sum[entity] += news.entity_occur[entity]
                            period_sentiment_med[entity].append(news.entity_sentiment[entity])
                            period_sentiment_sum[entity] += news.entity_sentiment[entity]*news.entity_occur[entity]
            i += 1

    date_time_series = pd.Series(date_time_series)
    count_time_series = pd.DataFrame(aggregate_count_sum).dropna(how='all')
    count_time_series["Time"] = date_time_series
    count_time_series.set_index('Time', inplace=True)
    count_time_series.index = pd.to_datetime(count_time_series.index)

    sentiment_med_time_series = pd.DataFrame(aggregate_sentiment_med).dropna(how='all')
    sentiment_med_time_series["Time"] = date_time_series
    sentiment_med_time_series.set_index('Time', inplace=True)
    sentiment_med_time_series.index = pd.to_datetime(sentiment_med_time_series.index)
    # print(sentiment_med_time_series)

    sentiment_sum_time_series = pd.DataFrame(aggregate_sentiment_sum).dropna(how='all')
    sentiment_sum_time_series["Time"] = date_time_series
    sentiment_sum_time_series.set_index('Time', inplace=True)
    sentiment_sum_time_series.index = pd.to_datetime(sentiment_sum_time_series.index)

    if rolling:
        sentiment_sum_time_series = sentiment_sum_time_series.ewm(alpha=rolling_smoothing_factor).mean()
        count_time_series = count_time_series.rolling(window=period).sum()
        sentiment_med_time_series = sentiment_med_time_series.ewm(alpha=rolling_smoothing_factor).mean()
    return count_time_series.sort_index(), sentiment_med_time_series.sort_index(), sentiment_sum_time_series.sort_index()
