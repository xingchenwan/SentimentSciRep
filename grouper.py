# Grouper module: generate different classification of the entities, either based on dynamic cosine distance in relevant
# co-occurrence matrices or by static sector categorisation
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance


import community
from utilities import *
import numpy as np
import os


def cache_graphs(full_data_obj, network_variables, window_size=90):
    """Cache all the graphs on every trading day"""
    # dump = {}

    for d in full_data_obj.days:
        dt = d.date
        if os.path.exists('data/networks/cooccurence_net' + str(dt) + 'window_' + str(window_size) + '.pickle'):
            print('The pickle file of the current date' + str(dt) + ' is already found. Skipping.')
            continue
        elif dt - datetime.timedelta(days=window_size) <= full_data_obj.start_date:
            continue
        else:
            sub_obj = create_sub_obj(full_data_obj, dt - datetime.timedelta(days=window_size), dt)
            G = sub_obj.build_occurrence_network_graph(focus_iterable=network_variables)
            # dump[str(dt)] = G
            print('Graph on date ' + str(dt) + ' Saved!')
            pickle.dump(G, open('data/networks/cooccurence_net' + str(dt) + 'window_' + str(window_size) + '.pickle',
                                "wb"))


def cache_nearest_neighbour(full_data_obj, names, window_size=90, save_path='data/nearest_neighbours/'):
    """
    Cache all the nearest neighbour computations
    Return in the format of
    {date1: {company1: [nearest_neighbours], company2: [nearest_neighbours] ... },
    date2: {company1: [nearest_neighbours], company2: [nearest_neighbours] ...}
    ...
    }
    :param save_path:
    :param names:
    :param save_path:
    :return:
    """
    result = {}
    for d in full_data_obj.days:
        dt = d.date
        result[str(dt)] = {}
        if dt - datetime.timedelta(days=window_size) <= full_data_obj.start_date:
            sub_obj = create_sub_obj(full_data_obj, full_data_obj.start_date, dt)
        else:
            sub_obj = create_sub_obj(full_data_obj, dt - datetime.timedelta(days=window_size), dt)
        G = sub_obj.build_occurrence_network_graph(focus_iterable=names)
        for name in names:
            update = get_nearest_neighbour([name], names, G)
            if update is not None:
                result[str(dt)].update(update)
        print(str(dt))  # , result[str(dt)])
    pickle.dump(result, open(save_path + "nearest_neighbours_" + str(window_size) + ".pickle", 'wb'))
    return result


def get_nearest_neighbour(names: list, network_variables,
                          G=None,
                          path=None,
                          full_data_obj=None,
                          load_from_cache=False,
                          start_date=None, end_date=None,
                          weight_threshold=0.,
                          quantile=0.,
                          top_n: int = 10):
    """
    Find the nearest neighbour of a name, or an iterable of names on the full data object constructed sentiment network
    Return in the format of:
    weight_threshold: set an absolute threshold for all names; neighbours with edge weight below this absolute threshold
    will be discarded.
    quantile: quantile threshold. e.g. 0.8 means only neighbours with weight
    {'Company A': [NN1, NN2, ... ],
    'Company B' : [NN3, NN4, ...]}

    By default, return the 10 nearest neighbours of each name of company requested.
    In this def,
    names refer to the companies where you *actually* want to compute the nearest neighbours, whereas network_variables
    refer to the companies to form the graph over which we compute nearest neighbours. Companies outside this second
    list will be discarded for graph purposes.
    """
    import pickle

    if G is None:
        if not load_from_cache:
            full_data_sub = create_sub_obj(full_data_obj, start_date, end_date)
            G = full_data_sub.build_occurrence_network_graph(focus_iterable=network_variables)
        else:
            if path:
                G = pickle.load(open(path, 'rb'))
            else:
                G = pickle.load(open('data/networks/cooccurence_net' + str(start_date) + 'window_90.pickle', 'rb'))
    res = {k: [] for k in names}
    threshold = {k: weight_threshold for k in names}
    if quantile or top_n:
        for name in names:
            if name not in list(G.nodes()):
                continue
            else:
                sub_graph = G[name]
                # Create a flat 1D array consisting of all the weights of the entity
                all_weights = np.array([v['weight'] for n, v in sub_graph.items()]).reshape(-1)
                # Compute the quantile weight threshold. E.g. the weight that corresponds to the 75th percentile if
                # quantile = 0.75
                thres = np.quantile(all_weights, quantile) if quantile else 0.
                if top_n:
                    sorted_weight = np.sort(all_weights, axis=None)[::-1]
                    thres = max(sorted_weight[min(top_n, sorted_weight.shape[0] - 1)], thres)
                threshold[name] = max(thres, threshold[name])

        for name in names:
            if name in list(G.nodes()):
                sub_graph = G[name]
                for n, v in sub_graph.items():
                    if v['weight'] > threshold[name]:
                        res[name].append(n)
    # pickle.dump(res, open('data/networks/NearestNeighbour_StartDate' + str(start_date) + "_EndDate" + str(end_date)
    #                       + "_" + ".pickle", "wb"))
    # print('Nearest Neighbour Saved!')
    return res


def get_dynamic_grouping(names, full_data_obj, start_date=None, end_date=None, save_to_csv=True, resolution=1.0):
    """
    Compute the dynamic grouping dictionary of a series of single names based on the cosine-distance of the entities
    within the specified period of time
    :param start_date: optional. If none, the start date will be taken to be the same as the starting date of the
    full_data object
    :param end_date: optional. Same as above
    :return: Dict mapping names to their dynamic grouping
    """
    if start_date and not isinstance(start_date, datetime.date):
        raise TypeError("Start time needs to be a date object")
    if end_date and not isinstance(end_date, datetime.date):
        raise TypeError("End time needs to be a date object")
    if not isinstance(full_data_obj, FullData):
        raise TypeError("full_data_obj needs to be a FullData object")
    if isinstance(names, pd.Series):
        names = names.values.tolist()
    start_date = start_date if start_date and start_date >= full_data_obj.start_date else full_data_obj.start_date
    end_date = end_date if end_date and end_date <= full_data_obj.end_date else full_data_obj.end_date
    # Sanity Checks

    full_data_sub = create_sub_obj(full_data_obj, start_date, end_date)
    G = full_data_sub.build_occurrence_network_graph(focus_iterable=names, )
    partition = community.best_partition(G, weight='weight', resolution=resolution)
    if save_to_csv:
        pd.Series(partition).to_csv('dynamic_grouping' + str(start_date) + str(end_date) + '.csv')
    return partition


def get_grouping_from_csv(path):
    data = pd.read_csv(path, header=None, )
    names = data.iloc[:, 0].values.tolist()
    group = data.iloc[:, 1].values.tolist()
    return dict(zip(names, group))


def get_sector_grouping(names, path, name_col, sector_col):
    """
    Get the ground truth static sector grouping dictionary of a series of single names. If there are any names in
    'names' argument that do not have a sector grouping in the lookup csv file, these names will be assigned an
    "unclassified" grouping in the output dictionary
    :param path: str. path string of the csv file containing the sector information
    :param name_col: str or int. If str, this argument is interpreted as the name of the column. If int,
    this arguement is interpreted as the column number
    :param sector_col: Same as above for the sector column
    :return: Dict mapping names to their sector grouping
    """
    data = pd.read_csv(path, header=None)
    if isinstance(name_col, str):
        assert name_col in list(data.columns)
        name_column = data[name_col]
    elif isinstance(name_col, int):
        assert name_col <= len(data.columns)
        name_column = data.iloc[:, name_col]
    else:
        raise TypeError("Invalid name column data type")

    if isinstance(sector_col, str):
        assert sector_col in list(data.columns)
        sector_column = data[sector_col]
    elif isinstance(sector_col, int):
        assert sector_col <= len(data.columns)
        sector_column = data.iloc[:, sector_col]
    else:
        raise TypeError("Invalid sector column data type")

    sector_lookup = pd.Series(sector_column.values, index=name_column).to_dict()
    # print("Names not found in the lookup:", [i for i in names if i not in sector_lookup.keys()])
    return {name: sector_lookup[name] if name in sector_lookup.keys() else 'Unclassified' for name in names}


def get_sector_peer(sector_dict, name):
    if not isinstance(sector_dict, dict):
        raise TypeError("sector_dict argument needs to be a dictionary.")
    if name not in sector_dict:
        raise ValueError("name not found in sector_dict")
    return [key for key, value in sector_dict.items() if value == sector_dict[name]]


def get_grouping_by_average_mkt_cap(path, sheet_name, division=4, focus_iterable=None):
    """Group the companies by the average market cap during the period of time in interest.
    division: number of groups to produce. e.g. division=4 means 4 groups consisting top 25%, next 25%, next 25% and
    bottom 25% will be produced.
    """
    res = {i: [] for i in range(division)}
    pct_sep_points = [0.] + [1. / i for i in range(division + 1, 1)]
    mkt_cap = pd.read_csv(path, sheet_name=sheet_name, index_col=0)
    if focus_iterable:
        focus_iterable = [x for x in mkt_cap.columns if x in focus_iterable]
        mkt_cap = mkt_cap[focus_iterable].abs()
    avg_mkt_cap = mkt_cap.mean(axis=0)
    # Generate the percentile rank of the market caps.
    avg_mkt_cap_rank = avg_mkt_cap.rank(pct=True)
    for i in range(len(pct_sep_points)):
        idx = avg_mkt_cap_rank[pct_sep_points[i] <= avg_mkt_cap_rank < pct_sep_points[i + 1]].index
        res[i] = avg_mkt_cap[idx]
    return res
