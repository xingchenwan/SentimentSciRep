# Xingchen Wan Oct 2019 MLRG, University of Oxford | xwan@robots.ox.ac.uk

import pandas as pd, numpy as np
import pickle
from collections import Counter
import datetime


def _get_z_score(frame, window):
    """
    Compute the z score used for the time series anomaly detection heuristic.
    This is indeed a very simple heuristic, and can of course be improved by a more advanced time series anomaly detec-
    tion regime. This is for example, prone to skewed distributions and window size needs be set manually.
    """
    r = frame.rolling(window=window)
    m = r.mean()
    s = r.std(ddof=0)
    return (frame-m)/s


def _get_event_series(z_score_series, threshold, ignore_window):
    """From the z_score_series, generate an event series which has the same length as the z-score series. The event
    series has three possible value. +1 for a positive sentiment event, 0 for non-event and -1 for negative sentiment
    event.
    :param threshold: the amount of standard deviaation. (Default +- 2.5 /sigma)
    :param ignore_window: the number of days after a detected event, up to which any new event detected will be ignored.
    Can be thought of a 'cool-off' period.
    """
    event_series = pd.Series(0, index=z_score_series.index)
    ignore_cnt = 0
    for i in range(len(z_score_series)):
        if ignore_cnt != 0:
            ignore_cnt -= 1
        pt = z_score_series[i]
        if pt > threshold: status = 1
        elif pt < -threshold: status = -1
        else: status = 0

        if ignore_cnt: status = 0
        if status != 0 and ignore_cnt == 0:
            ignore_cnt = ignore_window
        event_series.iloc[i] = status
    return event_series


def event_analyser(names: list,
                   sentiment_frame: pd.DataFrame,
                   rolling_window: int = 180,
                   detection_threshold: float = 2.5,
                   ignore_window: int = 10,
                   save_csv: bool = True,
                   start_date: str = None,
                   end_date: str = None,) -> pd.DataFrame:
    """
    Obtain the sentiment events of the companies
    :param names: the names of the company.
    :param sentiment_frame: the pandas DataFrame containing the sentiment time series of all said companies (and beyond)
    :param rolling_window: the number of days used as a heuristic for the simple sentiment event detection
    :param detection_threshold: the amount of standard deviation. (Default +- 2.5 /sigma)
    :param ignore_window:the number of days after a detected event, up to which any new event detected will be ignored.
    Can be thought of a 'cool-off' period after a detected event
    :param save_csv: whether save the result as a csv in the current directory.
    :param start_date: :param end_date: in string formatted as YYYYMMDD. If left blank all available data will be used
    in the computation
    :return: the dataframe of the same size as the sentiment frame, but with entries -1 (neg. sentiment events), 0 (
    non-event), 1 (pos. sentiment events).
    """

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > sentiment_frame.index[0] else sentiment_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < sentiment_frame.index[-1] else sentiment_frame.index[-1]
    sentiment_frame = sentiment_frame[(sentiment_frame.index >= start_date) & (sentiment_frame.index <= end_date)]

    res = {}
    i = 0
    for name in names:
        print("Progress: "+str(i)+" / "+str(len(names)))
        exogen_series = sentiment_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold, ignore_window)
        res[name] = event
        i += 1
    res = pd.DataFrame(res)
    if save_csv:
        res.to_csv("sentiment_events.csv")
    return res


def fix_fulldata(full_data):
    """
    A temporary patch to resolve the incompatibility issue between Python 2.x and 3.x pickling
    :param full_data:
    :return:
    """

    def _fix_object(obj):
        """
        Decode the byte data type back to strings due to Python 3.x un-pickling
        :param obj: all_data object
        :return: None
        """
        obj.__dict__ = dict((k.decode("utf8"), v) for k, v in obj.__dict__.items())

    def _fix_iterables(obj):
        """
        Decode iterables
        :param obj: An iterable, dict or list (other iterables are not included as they were not included in the
        original data type design
        :return: Fixed iterables
        """
        if isinstance(obj, list):
            return [x.decode('utf8') if isinstance(x, bytes) else x for x in obj]
        elif isinstance(obj, dict):
            return {k.decode('utf8'): v for k, v in obj.items()}

    _fix_object(full_data)
    full_data.entity_occur_interval = _fix_iterables(full_data.entity_occur_interval)
    full_data.entity_sentiment_interval = _fix_iterables(full_data.entity_sentiment_interval)

    for day in full_data.days:
        _fix_object(day)
        day.entity_sentiment_day = _fix_iterables(day.entity_sentiment_day)
        day.entity_occur_day = _fix_iterables(day.entity_occur_day)
        for news in day.day_news:
            _fix_object(news)
            news.entity_occur = _fix_iterables(news.entity_occur)
            news.entity_sentiment = _fix_iterables(news.entity_sentiment)


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
    :param rolling: whether to use rolling average
    :param rolling_smoothing_factor: the smoothing factor used for exponentially weighted moving average
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
    return count_time_series, sentiment_med_time_series, sentiment_sum_time_series


if __name__ == "__main__":

    # Changes these accordingly please.
    MARKET_DATA = 'data/market_data.xlsx'
    SHEET_NAME = 'Price1'
    FULL_DATA = 'data/full.date.20061020-20131120'

    data = pd.read_excel(MARKET_DATA, sheet_name=SHEET_NAME, index_col=0).astype('float64')
    all_names = list(data.columns)[:-1]

    with open(FULL_DATA, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    fix_fulldata(all_data)
    print("Preparing sentiment time series... ")
    count, med_sentiment, sum_sentiment = process_count_sentiment(all_data, rolling=True, rolling_smoothing_factor=.7, focus_iterable=all_names)#
    print("Generating event time series...")
    event_analyser(all_names, sum_sentiment, 180, 2.5, 10, True,)