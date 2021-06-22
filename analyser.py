# Analyser: modules that perform numerical analyses on the result
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
from processor import process_beta_time_series
from utilities import *


def _get_z_score(frame, window):
    r = frame.rolling(window=window)
    m = r.mean()
    s = r.std(ddof=0)
    # plt.plot((frame-m)/s)
    # plt.show()
    return (frame - m) / s


def _get_event_series(z_score_series, threshold, ignore_window=False):
    event_series = pd.Series(0, index=z_score_series.index)
    # ignore_cnt = 0
    for i in range(len(z_score_series)):
        # if ignore_cnt != 0:
        #    ignore_cnt -= 1
        pt = z_score_series[i]
        if pt > threshold:
            status = 1
        elif pt < -threshold:
            status = -1
        else:
            status = 0

        # if ignore_cnt: status = 0
        # if status != 0 and ignore_cnt == 0:
        #    ignore_cnt = ignore_window
        event_series.iloc[i] = status
    return event_series


def get_abnormal_return(name, log_return_frame, date, window, beta_path='data/^GSPC.csv'):
    # The normal return is given by the Market Model over MSCI return over the last window
    open_date = date - pd.to_timedelta(window, 'D')
    hist_returns = log_return_frame[['_ALL', name]]
    hist_returns = hist_returns[(hist_returns.index >= open_date) & (hist_returns.index < date)]

    mkt_return_today = log_return_frame.loc[date, '_ALL']
    stock_return_today = log_return_frame.loc[date, name]

    # Estimate alpha and beta using simple OLS
    X = hist_returns['_ALL'].values
    y = hist_returns[name].values
    beta, alpha, r_val, p_val, std_err = st.linregress(X, y)
    # disturb_var = 1/(window - 2)*np.sum(np.square(y - alpha - beta*X))
    # print(beta)
    abnormal_return = stock_return_today - alpha - beta * mkt_return_today
    return abnormal_return


def get_event_cumulative_abnormal_return(name, event_series, z_series, daily_abnormal_return_series, max_lag):
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(0, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate', 'Type'] + day_range)
    if z_series is not None:
        z_res = pd.DataFrame(0, index=res.index, columns=res.columns)
    else:
        z_res = None
    i = 0
    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:
            dt = event_series.index[day]
            res.loc[i, 'Name'] = name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            if z_series is not None:
                z_res.loc[i, 'Name'] = name
                z_res.loc[i, 'EventDate'] = dt
                z_res.loc[i, 'Type'] = event_series.iloc[day]

            cum_rtn = 0
            for lag in range(-max_lag, max_lag + 1):
                try:
                    cum_rtn += daily_abnormal_return_series[dt + BDay(lag)]
                except KeyError:
                    pass
                res.loc[i, lag] = cum_rtn
                if z_series is not None:
                    try:
                        z_res.loc[i, lag] = z_series[dt + BDay(lag)]
                    except KeyError:
                        z_res.loc[i, lag] = np.nan
            i += 1
    return res, z_res


def get_event_vol(name, event_series, z_series, vol_series, max_lag):
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] + day_range)
    z_res = pd.DataFrame(np.nan, index=res.index, columns=res.columns)
    i = 0
    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:
            dt = event_series.index[day]
            res.loc[i, 'Name'] = name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            # z_res.loc[i, 'Name'] = name
            # z_res.loc[i, 'EventDate'] = dt
            # z_res.loc[i, 'Type'] = event_series.iloc[day]

            for lag in day_range:
                try:
                    res.loc[i, lag] = vol_series[dt + BDay(lag)] if vol_series[dt + BDay(lag)] != 0 else np.nan
                except KeyError:
                    res.loc[i, lag] = np.nan
            #    try:
            #       z_res.loc[i, lag] = z_series[dt+BDay(lag)]
            #    except KeyError:
            #        z_res.loc[i, lag] = np.nan
            i += 1
    return res, z_res


def get_sentiment_change(name, sentiment_frame, start_date, current_date):
    try:
        sentiment_series = sentiment_frame[[name]]
    except KeyError:
        print(name + " not in sentiment frame.")
        return np.nan
    sentiment_series = sentiment_series[
        (sentiment_series.index >= start_date) & (sentiment_series.index <= current_date)]
    # print(type(sentiment_series.index), type(current_date))
    if current_date == start_date:
        sentiment_change = 0
    elif current_date > start_date:
        try:
            sentiment_change = (sentiment_series.loc[current_date] - sentiment_series.loc[start_date]).item()
        except KeyError:
            return np.nan
    else:
        raise ValueError("Current_date must be after or on the same date as the start_date!")
    # print(sentiment_change)
    return sentiment_change


def get_nearest_neighbours_sentiment(leading_name, event_series, nn_cache: dict, sentiment_frame, max_lag=7):
    """
    Get the sentiment diffusion from one event to the group average sentiment
    """
    import datetime
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] + day_range)
    i = 0

    # Get the start and end dates of the nn_cache; ignore queries outside this range
    cache_keys = list(nn_cache.keys())
    cache_start_date, cache_end_date = min(cache_keys), max(cache_keys)
    cache_start_date = datetime.datetime.fromisoformat(cache_start_date)
    cache_end_date = datetime.datetime.fromisoformat(cache_end_date)

    for day in range(len(event_series)):
        # Load the nearest neighbour of the leading name *on that day*

        if event_series.iloc[day] != 0:  # A non-zero entry signifies a sentiment event
            dt = event_series.index[day]  # Get the date
            if dt < cache_start_date or dt > cache_end_date:
                continue
            else:
                try:
                    lagging_names = nn_cache[str(dt.date())][leading_name]
                except KeyError:
                    return None
                res.loc[i, 'Name'] = leading_name
                res.loc[i, 'EventDate'] = dt
                res.loc[i, 'Type'] = event_series.iloc[day]

                for lag in day_range:
                    cumulative_sentiment = []
                    for name in lagging_names:
                        if name != leading_name:
                            cumulative_sentiment.append(
                                get_sentiment_change(name, sentiment_frame, dt - BDay(max_lag), dt + BDay(lag)))
                    res.loc[i, lag] = np.nanmean(
                        np.array(cumulative_sentiment))  # Get the mean accumulative change in sentiment across the group
                i += 1
    return res


def get_nearest_neighbours_price(leading_name, event_series, nn_cache: dict, px_frame, max_lag=7,
                                 px_series='rtn'):
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'NN', 'EventDate', 'Type'] + day_range)

    cache_keys = list(nn_cache.keys())
    cache_start_date, cache_end_date = min(cache_keys), max(cache_keys)
    cache_start_date = datetime.datetime.fromisoformat(cache_start_date)
    cache_end_date = datetime.datetime.fromisoformat(cache_end_date)

    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:  # A non-zero entry signifies a sentiment event
            dt = event_series.index[day]  # Get the date
            if dt < cache_start_date or dt > cache_end_date:
                continue
            else:
                # Get the nearest neighbours on that date
                lagging_names = nn_cache[str(dt.date())][leading_name]
                for n in lagging_names:
                    this_res = {'Name': leading_name, 'NN': n, "EventDate": str(dt.date()), "Type": event_series.iloc[day]}
                    for lag in day_range:
                        cum_rtn = 0.
                        if px_series == 'rtn':
                            try:
                                cum_rtn += get_abnormal_return(n, px_frame, dt + BDay(lag), 180)
                                this_res.update({lag: cum_rtn})
                            except KeyError:
                                this_res.update({lag: np.nan})
                        else: # vol mode
                            try:
                                vol = px_frame.loc[dt + BDay(lag), n] if px_frame.loc[dt + BDay(lag), n] != 0 else np.nan
                            except KeyError:
                                vol = np.nan
                            this_res.update({lag: vol})
                    this_res = pd.Series(this_res)
                    res = res.append(this_res, ignore_index=True)
    return res


def get_group_sentiment(leading_name, lagging_names, event_series, sentiment_frame, max_lag, ):
    """
    Get the sentiment diffusion from one event to the group average sentiment
    """
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] + day_range)
    i = 0
    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:  # A non-zero entry signifies a sentiment event
            dt = event_series.index[day]  # Get the date
            res.loc[i, 'Name'] = leading_name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            for lag in day_range:
                cumulative_sentiment = []
                for name in lagging_names:
                    if name != leading_name:
                        cumulative_sentiment.append(
                            get_sentiment_change(name, sentiment_frame, dt - BDay(max_lag), dt + BDay(lag)))
                res.loc[i, lag] = np.nanmean(
                    np.array(cumulative_sentiment))  # Get the mean accumulative change in sentiment across the group
            i += 1
    # print(res)
    return res


def get_group_px(leading_name, lagging_names, event_series, market_frame, max_lag, mode='vol'):
    """Get the average change in price/volatility etc from on event to group average"""
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] + day_range)
    i = 0
    assert mode in ['vol', 'rtn'], "invalid mode " + str(mode)

    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:
            dt = event_series.index[day]  # Get the date
            res.loc[i, 'Name'] = leading_name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            for lag in day_range:
                cumulative_px = []
                cumulative_rtn = 0.
                for name in lagging_names:
                    if name != leading_name:
                        if mode == 'rtn':
                            try:
                                cumulative_rtn += get_abnormal_return(name, market_frame, dt + BDay(lag), 180)
                            except KeyError:  # Some days do not have return, for some reason
                                pass
                            cumulative_px.append(cumulative_rtn)
                        else:
                            try:
                                cumulative_px.append(market_frame.loc[dt + BDay(lag), name])
                            except KeyError:
                                cumulative_px.append(np.nan)
                res.loc[i, lag] = np.nanmean(np.array(cumulative_px))
                # Get the mean accumulative change in sentiment across the group
            i += 1
    return res


def get_avg_col(names, market_frame):
    # Get the average volatility of a single name / a collection of names (e.g. all companies within a sector
    res = pd.Series(np.nan, index=names)
    for name in names:
        # Retrieve all volatility
        res[name] = np.nanmean(market_frame[name])
    all_average = np.nanmean(res)
    print("Average Vol", all_average)
    return res, all_average


def group_average(names, market_frame, sentiment_frame, rolling_window=180, detection_threshold=2.5, max_lag=7,
                  save_csv=True, mode='rtn', start_date=None, end_date=None, file_name=None, save_full_result=True,
                  wa='volume', smooth_weight=False, volume_frame=None, market_cap_frame=None, exclude_large_cap=0.,
                  aggregate_price=True):
    """
    Compute the sentiment impact on the price, all at the group level.
    note that by default, we use the SumSentiment, so at the group level we simply aggregate all the sentiment score of
    each individual company.
    For the group average return/volatility, we allow for different possible options:
     - Simple average
     - Volume-weighted average
     - Market-cap weighted average
     For the latter two methods, we additionally and optionally apply a 20 days smoothing window, as it is the convention
     to apply smoothing to these quantities
    :param names: a list of companies that are the constituents of the group
    :param market_frame: a pd.DataFrame, containing the [px, volatility] data of all the companies
    :param sentiment_frame: a pd.DataFrame, containing [no. of mentions, sum sentiment, average sentiment] data of all
    the companies
    :param rolling_window: the rolling window to compute {alpha, beta} that are used to compute the abnormal return.
    default is 180 trading days
    :param detection_threshold: The standard deviation based detection threshold for the sentiment event. default: 2
    (i.e., if the sentiment score of company X at time t exceeds 2 standard deviation of the previous 180 days, we
    think the company experiences a *sentiment event*)
    :param max_lag: maximum lag/trailing window around the sentiment events. default 7: i.e. compute the changes in
    {price, volatility} in the 7 days before and after the sentiment events
    :param save_csv: bool. Whether save the results as a csv file. Default: True
    :param mode: {'rtn', 'vol'}: whether use *ReTurN' or 'VOLatility' mode computation. Note that return mode will cap-
    ture both the direction and magnitude of any changes in market data, where as volatility mode computes the magnitude
    only.
    :param start_date :param end_date: the start and end dates. If supplied, all data outside these date ranges will be
    discarded
    :param file_name:
    :param save_full_result:
    :param wa: *W*eighted *A*verage
    :param smooth_weight:
    :param volume_frame:
    :param market_cap_frame:
    :param exclude_large_cap:
    :return:
    """
    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > market_frame.index[0] \
        else market_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < market_frame.index[-1] else \
        market_frame.index[-1]
    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]
    sentiment_frame = sentiment_frame[(sentiment_frame.index >= start_date) & (sentiment_frame.index <= end_date)]
    if wa == 'volume' and volume_frame is None:
        raise ValueError("volume_wa is turned on but no volume_file is supplied.")
    elif wa == 'cap' and market_cap_frame is None:
        raise ValueError("market_cap_wa is turned on but no market_cap_frame is supplied.")
    if exclude_large_cap:
        if market_cap_frame is None:
            raise ValueError(
                "A Non-0 exclude_large_cap option is supplied, but no market_cap_frame information is supplied!")
        large_cap = get_big_cap_names(names, market_cap_frame, 0.1)

    if wa:
        if wa == "volume":
            names = [n for n in names if n in volume_frame.columns]
            weight = volume_frame[names]
        elif wa == 'cap':
            names = [n for n in names if n in market_cap_frame.columns]
            weight = market_cap_frame[names]
        if smooth_weight:
            weight = simple_moving_average(weight)
        weight = weight[(weight.index >= start_date) & (weight.index <= end_date)]

    # Compute the group SUM sentiment
    # if wa:
    #    sentiment_frame = sentiment_frame[names] * weight[names]
    #    for name in names:
    #        sentiment_frame[name] /= sum_volume
    #        all_nan_columns = sentiment_frame.columns[sentiment_frame.isnull().all(0)]
    #        names = [name for name in names if name not in all_nan_columns]
    #        sentiment_frame.dropna(inplace=True, axis=1, how='all')
    #        sentiment_frame.dropna(inplace=True)
    #    sentiment_frame = sentiment_frame[names].sum(axis=1)
    # else:

    sentiment_frame = sentiment_frame[names].sum(axis=1)
    z_score = _get_z_score(sentiment_frame, rolling_window)
    event = _get_event_series(z_score, detection_threshold)

    # The sum of weights
    sum_weight = pd.Series(0., index=market_frame.index)

    if aggregate_price:
        if mode == 'rtn':
            # Compute the group AVERAGE abnormal return
            daily_abnormal_return = pd.Series(0, index=market_frame.index, )
            for date in market_frame.index:
                if not date < start_date + pd.to_timedelta(rolling_window, "D") and date < end_date:
                    for name in names:
                        if exclude_large_cap and large_cap is not None and name in large_cap[date]: continue
                        if wa:
                            daily_abnormal_return[date] += get_abnormal_return(name, market_frame, date, window=180) \
                                                           * weight.loc[date, name] if np.isfinite(
                                get_abnormal_return(name, market_frame, date, window=180) \
                                * weight.loc[date, name]) else 0.
                            sum_weight.loc[date] += weight.loc[date, name] if np.isfinite(
                                weight.loc[date, name]) else 0.

                        else:
                            daily_abnormal_return[date] += get_abnormal_return(name, market_frame, date, window=180)
            if wa:
                daily_abnormal_return /= sum_weight
            res, _ = get_event_cumulative_abnormal_return('mean', event, z_score, daily_abnormal_return, max_lag)
        elif mode == 'vol':
            # Compute the group AVERAGE volatility
            avg_vol = pd.Series(0., index=market_frame.index)
            for date in market_frame.index:
                if not date < start_date + pd.to_timedelta(rolling_window, "D") and date < end_date:
                    for name in names:
                        if exclude_large_cap and large_cap is not None and name in large_cap[date]: continue
                        if wa:
                            avg_vol[date] += market_frame.loc[date, name] * weight.loc[date, name] \
                                if np.isfinite(market_frame.loc[date, name] * weight.loc[date, name]) else 0.
                            sum_weight.loc[date] += weight.loc[date, name] if np.isfinite(weight.loc[date, name]) else 0
                        else:
                            avg_vol[date] += market_frame.loc[date, name] if np.isfinite(
                                market_frame.loc[date, name]) else 0
            if wa:
                avg_vol /= sum_weight
            res, _ = get_event_vol('mean', event, None, avg_vol, max_lag)
        else:
            raise ValueError("Key" + str(mode) + " cannot be understood.")

    else:  # Do not aggregate the price - i.e. no averaging or whatever and save t
        day_range = [i for i in range(-max_lag, max_lag + 1)]
        res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_range)
        for name in names:
            if mode == 'rtn':
                daily_abnormal_return = pd.Series(np.nan, index=market_frame.index)
                for date in market_frame.index:
                    if date < start_date + pd.to_timedelta(rolling_window, 'D'):
                        continue
                    elif date > end_date:
                        break
                    daily_abnormal_return[date] = get_abnormal_return(name, market_frame, date, window=180)
                this_res, this_z_res = get_event_cumulative_abnormal_return(name, event, z_score,
                                                                            daily_abnormal_return, max_lag)
            else:
                this_res, this_z_res = get_event_vol(name, event, z_score, market_frame[name], max_lag)
            res = res.append(this_res, ignore_index=True)

    pos_res, neg_res = res[res['Type'] == 1], res[res['Type'] == -1]

    pos_summary = pos_res.describe()
    neg_summary = neg_res.describe()

    if save_csv:
        file_name = file_name if file_name is not None else str(names[0])
        if wa:
            file_name += '_vwap'
        if exclude_large_cap:
            file_name += '_excludeLargeCap_' + str(exclude_large_cap)
        if not aggregate_price:
            file_name += "_unaggregated"
        file_appendix = file_name + "_rollWindow" + str(rolling_window) + \
                        "_maxLag" + str(max_lag) + "_mode" + str(mode) + "_" + str(start_date.date()) + \
                        "_" + str(end_date.date()) + ".csv"
        if save_full_result:
            pos_res.to_csv('output/posFullEvents' + file_appendix)
            neg_res.to_csv("output/negFullEvents" + file_appendix)
        else:
            pos_summary.to_csv('output/posEvents' + file_appendix)
            neg_summary.to_csv('output/negEvents' + file_appendix)
    return pos_summary, neg_summary, pos_res, neg_res


def event_analyser(names, market_frame, exogen_frame, rolling_window=180, detection_threshold=2.5, max_lag=5,
                   save_csv=True, mode='rtn', start_date=None, end_date=None, file_name=None, save_full_result=True,
                   ):
    if mode == 'rtn':
        day_range = [i for i in range(-max_lag, max_lag + 1)]
    elif mode == 'vol':
        day_range = [i for i in range(-max_lag, max_lag + 1)]
    else:
        raise ValueError("Unrecognised mode argument: rtn or vol allowed")

    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_range)
    z_res = pd.DataFrame(columns=res.columns)

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > market_frame.index[0] else \
        market_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < market_frame.index[-1] else \
        market_frame.index[-1]
    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]
    exogen_frame = exogen_frame[(exogen_frame.index >= start_date) & (exogen_frame.index <= end_date)]

    for name in names:
        if mode == 'rtn':
            daily_abnormal_return = pd.Series(np.nan, index=market_frame.index)
            for date in market_frame.index:
                if date < start_date + pd.to_timedelta(rolling_window, 'D'):
                    continue
                elif date > end_date:
                    break
                daily_abnormal_return[date] = get_abnormal_return(name, market_frame, date, window=180)

        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)

        if mode == 'rtn':
            this_res, this_z_res = get_event_cumulative_abnormal_return(name, event, exogen_z_score,
                                                                        daily_abnormal_return, max_lag)
        else:
            this_res, this_z_res = get_event_vol(name, event, exogen_z_score, market_frame[name], max_lag)
        res = res.append(this_res, ignore_index=True)
        z_res = z_res.append(this_z_res, ignore_index=True)
    pos_res = res[res['Type'] == 1]
    neg_res = res[res['Type'] == -1]

    pos_z_score = z_res[z_res.Type == 1].mean(axis=0)
    pos_z_score.name = 'AvgZScore'
    neg_z_score = z_res[z_res.Type == -1].mean(axis=0)
    neg_z_score.name = 'AvgZScore'

    pos_summary_stat = pos_res.describe()
    neg_summary_stat = neg_res.describe()

    # Append t-test and Wilcoxon rank test p-values to the bottom of the result data frames
    pos_res_stat_tests = pos_res[day_range]
    neg_res_stat_tests = neg_res[day_range]

    # if mode is vol, we use the days leading up to the event day as the benchmark and conduct two-sample, one-tailed
    # t-test to ascertain whether the volatilities after the event days are significantly higher than before the event.
    if mode == 'vol':
        # T-test

        # Wilcoxon Rank-sum test
        # pos_wilcoxon = st.wilcoxon(pos_res_stat_tests.loc[:, list(range(0, max_lag))].values,
        #                           pos_res_stat_tests.loc[:, list(range(-max_lag, 0))].values,
        #                           alternative='greater') # one-tailed test here
        # neg_wilcoxon = st.wilcoxon(neg_res_stat_tests.loc[:, list(range(0, max_lag))].values,
        #                           neg_res_stat_tests.loc[:, list(range(-max_lag, 0))].values,
        #                           alternative='greater')  # one-tailed test here
        #

        # Convert two tailed p-value to one-tailed p-value
        pos_pval = pd.Series(1., name='t-test', index=pos_res_stat_tests.columns)
        neg_pval = pd.Series(1., name='t-test', index=neg_res_stat_tests.columns)

        for i in day_range:
            if i >= -int(max_lag / 2):
                pos_t_test = st.ttest_1samp(pos_res_stat_tests.loc[:, i].values,
                                            popmean=np.nanmean(pos_res_stat_tests.loc[:,
                                                               list(range(-max_lag, -int(max_lag / 2)))].values),
                                            nan_policy='omit')
                neg_t_test = st.ttest_1samp(neg_res_stat_tests.loc[:, i].values,
                                            popmean=np.nanmean(neg_res_stat_tests.loc[:,
                                                               list(range(-max_lag, -int(max_lag / 2)))].values),
                                            nan_policy='omit')

                pos_pval[i] = pos_t_test[1] / 2 if pos_t_test[0] > 0 else 1 - pos_t_test[1] / 2
                neg_pval[i] = neg_t_test[1] / 2 if neg_t_test[0] > 0 else 1 - neg_t_test[1] / 2

        pos_summary_stat = pos_summary_stat.append(pos_pval)
        # here we divide the p-value by 2 to obtain the one tailed p value

        # wilcoxon_pos = pd.Series({i: st.wilcoxon(pos_res_stat_tests[i] - pos_pop_mean)[1]
        #                          for i in day_range}, name='wilcoxon', index=pos_res_stat_tests.columns)

        neg_summary_stat = neg_summary_stat.append(neg_pval)

    else:
        # 2-tailed t-test against the hypothesis that the abnormal return should be 0
        pos_summary_stat = pos_summary_stat.append(pd.Series(
            st.ttest_1samp(pos_res_stat_tests, popmean=0, nan_policy='omit')[1], name='t-test',
            index=pos_res_stat_tests.columns))
        # wilcoxon_pos = pd.Series({i: st.wilcoxon(pos_res_stat_tests[i] - 0)[1]
        #                      for i in day_range}, name='wilcoxon', index=pos_res_stat_tests.columns)
        # pos_summary_stat = pos_summary_stat.append(wilcoxon_pos)
        # pos_summary_stat = pos_summary_stat.append(pos_z_score)
        neg_summary_stat = neg_summary_stat.append(pd.Series(
            st.ttest_1samp(neg_res_stat_tests, popmean=0, nan_policy='omit')[1], name='t-test',
            index=neg_res_stat_tests.columns))
        # wilcoxon_neg = pd.Series({i: st.wilcoxon(neg_res_stat_tests[i] - 0)[1] for i in day_range}, name='wilcoxon'
        #                      , index=neg_res_stat_tests.columns)
        # neg_summary_stat = neg_summary_stat.append(wilcoxon_neg)
        neg_summary_stat = neg_summary_stat.append(neg_z_score)

    if save_csv:
        file_name = file_name if file_name is not None else str(names[0])
        file_appendix = file_name + "_rollWindow" + str(rolling_window) + \
                        "_maxLag" + str(max_lag) + "_mode" + str(mode) + "_" + str(start_date.date()) + \
                        "_" + str(end_date.date()) + ".csv"
        if save_full_result:
            pos_res.to_csv('output/Individual_Price_Analyser_Output/posFullEvents' + file_appendix)
            neg_res.to_csv("output/Individual_Price_Analyser_Output/negFullEvents" + file_appendix)
        else:
            pos_summary_stat.to_csv('output/Individual_Price_Analyser_Output/posEvents' + file_appendix)
            neg_summary_stat.to_csv('output/Individual_Price_Analyser_Output/negEvents' + file_appendix)
    return pos_summary_stat, neg_summary_stat, pos_res, neg_res


def nearest_neighbour_sentiment_analyser(names, neighbours: dict, sentiment_frame, rolling_window=90,
                                         detection_threshold=2., max_lag=7, start_date=None,
                                         end_date=None, save_csv=True, file_name=None):
    """
    Compute the sentiment diffusion from a single name (or a list of names) to the their respective groups
    of nearest neighbours. neighbours should be a list of list, with shape[0] matching the length of the list
    of names passed. Other arguments are as usual.

    Added by Xingchen on 27 Jan 2020
    """
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_range)
    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > sentiment_frame.index[0] else \
        sentiment_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < sentiment_frame.index[-1] else \
        sentiment_frame.index[-1]
    exogen_frame = sentiment_frame[(sentiment_frame.index >= start_date) & (sentiment_frame.index <= end_date)]

    c = 1
    for name in names:
        print('Processing ' + str(c) + "/" + str(len(names)))
        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)
        this_res = get_group_sentiment(name, neighbours[name], event, exogen_frame, max_lag)
        res = res.append(this_res,ignore_index=True)
        c += 1

    pos_res, neg_res = res[res['Type'] == 1], res[res['Type'] == -1]
    if save_csv:
        file_name = file_name if file_name is not None else str(names[0])
        file_appendix = file_name + "_Sentiment_rollWindow" + str(rolling_window) + \
                        "_maxLag" + str(max_lag) + "_mode" + "_" + str(start_date.date()) + \
                        "_" + str(end_date.date()) + ".csv"
        pos_res.to_csv("output/NearestNeighbour_Sentiment_Analyser_Output/posFullEvents" + file_appendix)
        neg_res.to_csv("output/NearestNeighbour_Sentiment_Analyser_Output/negFullEvents" + file_appendix)


def group_sentiment_analyser(names, sentiment_frame, rolling_window=90, lagging_names=None,
                             detection_threshold=2, max_lag=5, start_date=None, end_date=None, save_csv=False,
                             file_name=None, save_full_result=True):
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_range)

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > sentiment_frame.index[0] else \
        sentiment_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < sentiment_frame.index[-1] else \
        sentiment_frame.index[-1]
    exogen_frame = sentiment_frame[(sentiment_frame.index >= start_date) & (sentiment_frame.index <= end_date)]
    if lagging_names is None: lagging_names = names

    for name in names:
        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)

        this_res = get_group_sentiment(name, lagging_names, event, exogen_frame, max_lag)
        res = res.append(this_res, ignore_index=True)

    pos_res, neg_res = res[res['Type'] == 1], res[res['Type'] == -1]
    pos_summary_stat, neg_summary_stat = pos_res.describe(), neg_res.describe()

    if save_csv:
        file_name = file_name if file_name is not None else str(names[0])
        file_appendix = file_name + "_Sentiment_rollWindow" + str(rolling_window) + \
                        "_maxLag" + str(max_lag) + "_mode" + "_" + str(start_date.date()) + \
                        "_" + str(end_date.date()) + ".csv"
        if save_full_result:
            pos_res.to_csv("output/Group_Sentiment_Analyser_Output/posFullEvents" + file_appendix)
            neg_res.to_csv("output/Group_Sentiment_Analyser_Output/negFullEvents" + file_appendix)
        else:
            pos_summary_stat.to_csv('output/Group_Sentiment_Analyser_Output/posEvents' + file_appendix)
            neg_summary_stat.to_csv('output/Group_Sentiment_Analyser_Output/negEvents' + file_appendix)

    return pos_summary_stat, neg_summary_stat, pos_res, neg_res


def group_px_analyser(names, sentiment_frame, market_frame, rolling_window=90, lagging_names=None,
                      detection_threshold=2.5, max_lag=5, start_date=None, end_date=None, save_csv=False,
                      file_name=None, save_full_result=True,
                      mode='rtn'):
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_range)

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > sentiment_frame.index[0] else \
        sentiment_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < sentiment_frame.index[-1] else \
        sentiment_frame.index[-1]
    exogen_frame = sentiment_frame[(sentiment_frame.index >= start_date) & (sentiment_frame.index <= end_date)]
    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]

    if lagging_names is None: lagging_names = names
    j = 0
    for name in names:
        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)
        this_res = get_group_px(name, lagging_names, event, market_frame, max_lag, mode=mode)
        res = res.append(this_res, ignore_index=True)
        j += 1
        print("Progress: " + str(j) + "/" + str(len(names)))
    pos_res, neg_res = res[res['Type'] == 1], res[res['Type'] == -1]
    pos_summary_stat, neg_summary_stat = pos_res.describe(), neg_res.describe()

    if save_csv:
        file_name = file_name if file_name is not None else str(names[0])
        file_appendix = "pxDiffusion_" + file_name + "_" + mode + "_rollWindow" + str(rolling_window) + \
                        "_maxLag" + str(max_lag) + "_mode" + "_" + str(start_date.date()) + \
                        "_" + str(end_date.date()) + ".csv"
        if save_full_result:
            pos_res.to_csv("Output/Group_Price_Analyser_Output/posFullEvents" + file_appendix)
            neg_res.to_csv("Output/Group_Price_Analyser_Output/negFullEvents" + file_appendix)
        else:
            pos_summary_stat.to_csv('output/Group_Price_Analyser_Output/posEvents' + file_appendix)
            neg_summary_stat.to_csv('output/Group_Price_Analyser_Output/negEvents' + file_appendix)

    return pos_summary_stat, neg_summary_stat, pos_res, neg_res


def network_event_analyser(leading_name, lagging_names, market_frame, exogen_frame, rolling_window=90,
                           detection_threshold=2.5, max_lag=5, mode='rtn', start_date=None, end_date=None):
    def process_sub_result(df, z_df):
        # print(z_df)
        groupby = df.groupby('Name')
        groupby_z = z_df.groupby('Name')
        res = {}
        for name, frame in groupby:
            data_stat_test = frame[day_rng]
            mean_z = groupby_z[day_rng].get_group(name).mean(axis=0)
            mean_z.name = 'AvgZScore'
            summary_stat = frame.describe()
            pop_mean = np.nanmean(data_stat_test.loc[:, list(range(-2 * max_lag, -1))].values) if mode == 'vol' else 0
            ttest = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[1], name='t-test',
                              index=day_rng)
            wilcoxon = pd.Series({i: st.wilcoxon(data_stat_test[i] - pop_mean)[1] for i in day_rng}, name='wilcoxon',
                                 index=day_rng)
            summary_stat = summary_stat.append(ttest)
            summary_stat = summary_stat.append(wilcoxon)
            # print(mean_z)
            summary_stat = summary_stat.append(mean_z)
            res[name] = summary_stat

        # Average exclude leading name:
        df = df[df.Name != leading_name]
        data_stat_test = df[day_rng]
        mean_z = data_stat_test.mean(axis=0)
        mean_z.name = 'AvgZScore'
        summary_stat = df.describe()

        pop_mean = np.nanmean(data_stat_test.loc[:, list(range(-2 * max_lag, -1))].values) if mode == 'vol' else 0
        ttest = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[1], name='t-test',
                          index=day_rng)
        tstats = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[0], name='t-stat',
                           index=day_rng)
        wilcoxon = pd.Series({i: st.wilcoxon(data_stat_test[i] - pop_mean)[1] for i in day_rng}, name='wilcoxon',
                             index=day_rng)
        wstats = pd.Series({i: st.wilcoxon(data_stat_test[i] - pop_mean)[0] for i in day_rng}, name='wstats',
                           index=day_rng)
        summary_stat = summary_stat.append(ttest)
        summary_stat = summary_stat.append(wilcoxon)
        summary_stat = summary_stat.append(mean_z)
        summary_stat = summary_stat.append(tstats)
        summary_stat = summary_stat.append(wstats)

        res['Average'] = summary_stat
        return res

    if mode == 'rtn':
        day_rng = [i for i in range(-max_lag, max_lag + 1)]
    else:
        day_rng = [i for i in range(-2 * max_lag, max_lag)]

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > market_frame.index[0] else \
        market_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < market_frame.index[-1] else \
        market_frame.index[-1]

    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]
    exogen_frame = exogen_frame[(exogen_frame.index >= start_date) & (exogen_frame.index <= end_date)]

    exogen_series = exogen_frame[leading_name]
    exogen_z_score = _get_z_score(exogen_series, rolling_window)
    event = _get_event_series(exogen_z_score, detection_threshold)
    # Get the sentiment event for the LEADING NAME
    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_rng)
    z_res = pd.DataFrame(columns=res.columns)

    # print(market_frame)
    # print(exogen_frame)

    for lagging_name in lagging_names:
        daily_abnormal_return = pd.Series(np.nan, index=market_frame.index)
        z_score = _get_z_score(exogen_frame[lagging_name], rolling_window)
        if mode == 'rtn':
            for date in market_frame.index:
                if date < start_date + pd.to_timedelta(rolling_window, 'D'):
                    continue
                daily_abnormal_return[date] = get_abnormal_return(lagging_name, market_frame, date, window=180)
            this_res, this_z_res = get_event_cumulative_abnormal_return(lagging_name, event, z_score,
                                                                        daily_abnormal_return, max_lag)
        else:
            this_res, this_z_res = get_event_vol(lagging_name, event, z_score, market_frame[lagging_name], max_lag)
        res = res.append(this_res, ignore_index=True)
        # print(this_z_res)
        z_res = z_res.append(this_z_res, ignore_index=True)

    pos = res[res['Type'] == 1]
    neg = res[res['Type'] == -1]
    pos_z = z_res[z_res.Type == 1]
    neg_z = z_res[z_res.Type == -1]
    print(pos)
    print(neg)

    pos_res = process_sub_result(pos, pos_z)
    neg_res = process_sub_result(neg, neg_z)
    print(pos_res)
    print(neg_res)

    return pos_res, neg_res
