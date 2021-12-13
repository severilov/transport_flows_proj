import json
import pickle

from scanf import scanf
import re
import numpy as np
import pandas as pd
import transport_graph as tg
import copy

import conf
from model import Model

pd.set_option('display.max_columns', None)

TNTP_TRIPS_FNAME = '../data/Custom_trips.tntp'  # '../data/SiouxFalls_trips.tntp'
TNTP_NET_FNAME = '../data/Custom_trips.tntp'  # '../data/SiouxFalls_net.tntp'

    
def parse_data(data_name: str='vladik') -> tuple:
    """
    Wrapper for different net and correspondence parsers.

    Parameters:
    -----------
    parser_name: str
        Name of transport net and corresponding parser.
    
    Returns:
    --------
    graph_data: dict
        See more in `GetGraphData`.
    L_dict, W_dict: dicts
        See more in `vladik_corr_parser` and such.
    """
    net_parser = None
    corr_parser = None
    if data_name == 'vladik':
        net_parser = vladik_net_parser
        corr_parser = vladik_corr_parser
    elif data_name == 'tntpd':
        net_parser = tntp_net_parser
        corr_parser = tntp_corr_parser
    elif data_name == 'custom':
        net_parser = custom_net_parser
        corr_parser = custom_corr_parser

    graph_data = GetGraphData(net_parser,
                                    columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])
    L_dict, W_dict = corr_parser()
    return graph_data, L_dict, W_dict


def vladik_net_parser() -> tuple:
    """
    Read `data/vl_links.txt` and `data/vl_nodes.txt` files containing
    Vladivostok transport network data. By default, reads `_test` versions 
    of these files.
    
    Returns:
    --------
    df: pd.DataFrame
        Contains following columns:
            init_node
            term_node
            capacity
            free_flow_time
            ['xa', 'xb', 'ya', 'yb'] - geographic coordinates of ``init_node`` and ``term_node``
    nodes: pd.DataFrame
        Contains following columns:
            node
            ['x', 'y', 'z'] - geographic coordinates of nodes.
    first_thru_node: int
        IDK what is this, but it is used later when creating columns 
        ``init_node_thru`` and ``term_node_thru``.
    """
    links = pd.read_csv(conf.vl_links_file, sep='\t', skiprows=0)

    # process pairs of nodes in one order
    links_ab = links[['ANODE', 'BNODE', 'cap_ab']].copy()
    links_ab.columns = ['init_node', 'term_node', 'capacity']
    links_ab['free_flow_time'] = (links.LENGTH / 1000) / links.speed_ab  # in hours
    # process pairs of nodes having capacities inversed
    links_ba = links[['ANODE', 'BNODE', 'cap_ba']].copy()
    links_ba.columns = ['init_node', 'term_node', 'capacity']
    links_ba['free_flow_time'] = (links.LENGTH / 1000) / links.speed_ba  # in hours
    df = links_ab.append(links_ba, ignore_index=True)

    # pairs of nodes are interchanged, so the graph becomes undirected.
    df_inv = df.copy()
    df_inv.columns = ['term_node', 'init_node', 'capacity', 'free_flow_time']
    df = df.append(df_inv, ignore_index=True)
    df = df[df.capacity > 0]
    df.drop_duplicates(inplace=True)

    nodes = pd.read_csv('data/vl_nodes.txt', sep='\t', skiprows=0).set_index('node')
    xa, xb, ya, yb = [], [], [], []
    for i in df.index:
        an, bn = df.init_node[i], df.term_node[i]
        xa.append(nodes.x[an])
        xb.append(nodes.x[bn])
        ya.append(nodes.y[an])
        yb.append(nodes.y[bn])

    df['xa'], df['xb'], df['ya'], df['yb'] = xa, xb, ya, yb

    # graph_data['nodes number'] = scanf('<NUMBER OF NODES> %d', metadata)[0]

    return df, nodes, 1


def tntp_net_parser():
    metadata = ''
    with open(TNTP_NET_FNAME, 'r') as myfile:
        for index, line in enumerate(myfile):
            if re.search(r'^~', line) is not None:
                skip_lines = index + 1
                headlist = re.findall(r'[\w]+', line)
                break
            else:
                metadata += line
    graph_data = {}
    nn = scanf('<NUMBER OF NODES> %d', metadata)[0]
    nl = scanf('<NUMBER OF LINKS> %d', metadata)[0]
    nz = scanf('<NUMBER OF ZONES> %d', metadata)[0]
    first_thru_node = scanf('<FIRST THRU NODE> %d', metadata)[0]

    dtypes = {'init_node': np.int32, 'term_node': np.int32, 'capacity': np.float64, 'length': np.float64,
                'free_flow_time': np.float64, 'b': np.float64, 'power': np.float64, 'speed': np.float64,
                'toll': np.float64,
                'link_type': np.int32}
    df = pd.read_csv(TNTP_NET_FNAME, names=headlist, dtype=dtypes, skiprows=skip_lines, sep=r'[\s;]+',
                        engine='python',
                        index_col=False)
    nt = None
    return df, nt, first_thru_node


def custom_net_parser():

    # dtypes = {'init_node': np.int32, 'term_node': np.int32, 'capacity': np.float64, 'length': np.float64,
    #           'free_flow_time': np.float64, 'b': np.float64, 'power': np.float64, 'speed': np.float64,
    #           'toll': np.float64,
    #           'link_type': np.int32}

    data = {'free_flow_time': [1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            'init_node': [1, 1, 2, 2, 1, 3, 3, 2],
            'term_node': [2, 2, 1, 1, 3, 1, 2, 3],
            'capacity': [1999, 4000, 4000, 1999, 4000, 4000, 4000, 4000],
            'length': [1000, 2000, 2000, 1000, 2000, 2000, 2000, 2000],
            'b': [0, 0, 0, 0, 0, 0, 0, 0],
            'power': [0, 0, 0, 0, 0, 0, 0, 0],
            'speed': [0, 0, 0, 0, 0, 0, 0, 0],
            'toll': [0, 0, 0, 0, 0, 0, 0, 0],
            'link_type': [0, 0, 0, 0, 0, 0, 0, 0]}

    df = pd.DataFrame(data=data)
    nt = None
    first_thru_node = 1
    return df, nt, first_thru_node


def vladik_corr_parser() -> tuple:
    """
    Read `data/vl_trips.txt` file and get constraints on correspondencies. Поясню на русском - 
    получаем (l, w) из статьи, там это называется источники и стоки корреспонденций.

    Important remark: L_dict and W_dict have same set of keys.
    
    Returns:
    --------
    L_dict: dict(int: int)
        Dict with nodes as keys and corresponding l's.
    W_dict: dict(int: int)
        Dict with nodes as keys and corresponding w's.
    """
    with open(conf.vl_trips_file, 'r') as fin:
        fin = list(fin)[1:]
        nodes = [int(x) for x in fin[0].split(' ')]
        L = [int(x) for x in fin[1].split(' ')]
        W = [int(x) for x in fin[2].split(' ')]

    return dict(zip(nodes, L)), dict(zip(nodes, W))


def tntp_corr_parser():
    with open(TNTP_TRIPS_FNAME, 'r') as myfile:
        trips_data = myfile.read()

    total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]
    # zones_number = scanf('<NUMBER OF ZONES> %d', trips_data)[0]

    origins_data = re.findall(r'Origin[\s\d.:;]+', trips_data)

    L_dict, W_dict = {}, {}
    for data in origins_data:
        origin_index = scanf('Origin %d', data)[0]
        origin_correspondences = re.findall(r'[\d]+\s+:[\d.\s]+;', data)
        L_dict[origin_index] = 0
        for line in origin_correspondences:
            target, corrs = scanf('%d : %f', line)
            L_dict[origin_index] += corrs
            if target not in W_dict:
                W_dict[target] = 0
            W_dict[target] += corrs

    return L_dict, W_dict  # , od


def custom_corr_parser():

    total_od_flow = 12000

    L_dict = {1: 4000, 2: 4000, 3: 4000}
    W_dict = {1: 4000, 2: 4000, 3: 4000}

    return L_dict, W_dict


def GetGraphData(parser, columns: list) -> dict:
    """
    Get data from desired dataset and convert it to pd.Dataframe of desired format.
    
    Parameters:
    -----------
    parser: function
        One of `net_parsers` of this module.
    columns: list of str
        Which columns to leave in resulting dataset.
    
    Returns:
    --------
    graph_data: dict
        Contains 4 fields:
        `graph_table` - pd.DataFrame, data for transport graph;
        `nodes_number` - int, number of unique nodes in graph;
        `links_number` - int, number of links in graph;
        `nodes_table` - pd.DataFrame, nodes with their coordinates;
    """
    graph_data = {}
    gt, nt, first_thru_node = parser() # first_thru_node is 1
    gt = gt[columns]

    # following two lines of code I don't understand much
    # But, effectively 'init_node_thru' and 'term_node_thru' will all be True
    gt.insert(loc=gt.columns.get_loc('init_node') + 1, column='init_node_thru',
                value=(gt['init_node'] >= first_thru_node))
    gt.insert(loc=gt.columns.get_loc('term_node') + 1, column='term_node_thru',
                value=(gt['term_node'] >= first_thru_node))
    graph_data['graph_table'] = gt
    graph_data['nodes number'] = len(set(gt.init_node.values) | set(gt.term_node.values))
    graph_data['links number'] = gt.shape[0]
    graph_data['nodes_table'] = nt
    return graph_data


def save_input_data_to_res(graph_data: pd.DataFrame, L_dict: dict, W_dict: dict) -> None:
    """
    Save transport graph data and LW-dicts to the folder results/input_data.
    Returns nothing.
    
    Parameters:
    -----------
    graph_data: pd.DataFrame
        Transport graph data.
    L_dict: dict
        L-dict from one of `corr_parsers` of this module.
    W_dict: dict
        W-dict from one of `corr_parsers` of this module.

    Returns:
    --------
        None
    """
    root = 'results/input_data/'
    with open(root + 'graph_data.pickle', 'wb') as fp:
        pickle.dump(graph_data, fp)
    with open(root + 'L_dict.json', 'w') as fp:
        json.dump(L_dict, fp)
    with open(root + 'W_dict.json', 'w') as fp:
        json.dump(W_dict, fp)


def ReadAnswer(filename):
    with open(filename) as myfile:
        lines = myfile.readlines()
    lines = lines[1:]
    flows = []
    times = []
    for line in lines:
        _, _, flow, time = scanf('%d %d %f %f', line)
        flows.append(flow)
        times.append(time)
    return {'flows': flows, 'times': times}

#### Katia multi-stage methods

def create_C(df, n, column_name):
    C = np.full((n, n), np.nan, dtype=np.double)
    column_ind = df.columns.get_loc(column_name)

    for index, raw_data_line in df.iterrows():
        i, j = int(raw_data_line[0]) - 1, int(raw_data_line[1]) - 1

        C[i, j] = raw_data_line[column_ind]
    return C


def reindexed_empty_corr_matrix(corr_dict: dict) -> tuple:
    """
    Reindex nodes and return corresponding matrices and empty correspondence matrix.
    Receives as input dict containing init_nodes as keys. Every value of this dict is dict of form
    {`'targets'`: list(int)}, containing list of all term_nodes.

    Reindexed indices are in the range `[0, n]`, where `n` is number of unique nodes.
    
    Parameters:
    ----------
    corr_dict: dict(int: dict)
        Contains as values dicts of form {`'targets'`: list(int)}.
    
    Returns:
    --------
    empty_corr_matrix: np.ndarray
        `(n x n)` matrix of zeros, where `n` is number of unique nodes in init_nodes and term_nodes.
    old_to_new: dict(int: int)
        Mapping between old indexes of nodes and new.
    new_to_old: dict(int: int)
        Mapping between new indexes of nodes and old.
    """
    # sort to add determined behaviour
    indexes = sorted(list(set(corr_dict.keys()) | set(sum([d['targets'] for d in corr_dict.values()], []))))

    n = len(indexes)
    new_indexes = np.arange(n)
    old_to_new = dict(zip(indexes, new_indexes))
    new_to_old = dict(zip(new_indexes, indexes))
    empty_corr_matrix = np.zeros((n, n))

    return empty_corr_matrix, old_to_new, new_to_old


def corr_matrix_to_dict(corr_matrix: np.ndarray, new_to_old: dict) -> dict:
    """
    Given correspondence matrix and node mapping, transform it to dictionary format.

    Parameters:
    -----------
    corr_matrix: np.ndarray
        (n x n) correspondence matrix.
    new_to_old: dict
        Mapping from `[0, ..., n]` to original nodes indices.

    Returns:
    --------
    corr_dict: dict
        Keys are original nodes indices. Each value is dict with two keys:
            `'targets'` is array of original target nodes indices for key node;
            `'corrs'` is np.ndarray of correspondencies for key node.
    """
    corr_dict = {}
    n = np.shape(corr_matrix)[0]
    for i in range(n):
        source = new_to_old[i]
        corr_dict[source] = {}
        corr_dict[source]['targets'] = [new_to_old[x] for x in np.arange(n)]
        corr_dict[source]['corrs'] = corr_matrix[i]
    return corr_dict


def get_LW(L_dict: np.ndarray, W_dict: np.ndarray, new_to_old: dict) -> tuple:
    """
    Given L- and W-dicts and new nodes mapping, return L and W as np.ndarrays.
    Also returns people_num - total sum of L's.
    Resulting arrays contain on i'th place value for new_to_old[i] node.
    Also, L and W are divided by their means to satisfy constraints from paper.
    
    Parameters:
    ----------
    L_dict: dict(int: int)
        mapping between nodes and their l's
    W_dict: dict(int: int)
        mapping between nodes and their w's
    new_to_old: dict(int: int)
        mapping between old indices of nodes and new
    
    Returns:
    --------
    L: np.ndarray
        Normalized array of l values.
    w: np.ndarray
        Normalized array of w values.
    people_num: int
        Total sum of l's. Has the meaning of total number of people.
    """
    # reindex and turn to np.array
    L = np.array([L_dict[new_to_old[i]] for i in range(len(L_dict))], dtype=np.double)
    W = np.array([W_dict[new_to_old[i]] for i in range(len(W_dict))], dtype=np.double)
    people_num = L.sum()
    # normalize because in paper we have constraints sum(L) = 1, sum(W) = 1
    L /= np.nansum(L)
    W /= np.nansum(W)
    return L, W, people_num


def distributor_L_W(array):
    max_value = np.max(array)
    max_value_index = np.where(array == np.max(array))

    unique, counts = np.unique(array, return_counts=True)
    array_dict = dict(zip(unique, counts))
    # TODO remove exception
    try:
        zero_num = array_dict[0]
    except KeyError:
        print('this array without 0')
        return array
    array[max_value_index] = max_value - zero_num

    array[np.where(array == 0)[0]] = 1.0

    return array


def index_nodes(graph_table, graph_correspondences, fill_corrs=True):
    table = graph_table.copy()
    inits = np.unique(table['init_node'][table['init_node_thru'] == False])
    terms = np.unique(table['term_node'][table['term_node_thru'] == False])
    through_nodes = np.unique([table['init_node'][table['init_node_thru'] == True],
                                table['term_node'][table['term_node_thru'] == True]])

    nodes = np.concatenate((inits, through_nodes, terms))
    nodes_inds = list(zip(nodes, np.arange(len(nodes))))
    init_to_ind = dict(nodes_inds[: len(inits) + len(through_nodes)])
    term_to_ind = dict(nodes_inds[len(inits):])

    table['init_node'] = table['init_node'].map(init_to_ind)
    table['term_node'] = table['term_node'].map(term_to_ind)
    correspondences = {}
    for origin, dests in graph_correspondences.items():
        dests = copy.deepcopy(dests)
        d = {'targets': list(map(term_to_ind.get, dests['targets']))}
        if fill_corrs:
            d['corrs'] = dests['corrs']
        correspondences[init_to_ind[origin]] = d

    inds_to_nodes = dict(zip(range(len(nodes)), nodes))
    return inds_to_nodes, correspondences, table


def get_T_from_t(t: pd.Series, graph_data: dict, model: Model) -> dict:
    """
    Calculate travel times given t's and model.
    Честно говоря, я уже вообще не уверен, как это все работает, но
    тут видимо на вход подаются t отдельных ребер а выдаются времена T для корреспонденций.

    T_dict have original node indices.
    
    Parameters:
    -----------
    t: pd.Series
        Видимо, времена требующиеся на проезд по отдельным ребрам t_e.
    graph_data: dict
        Output of DataHandler.getGraphData, all data about transport graph.
    model: model.Model
        Model of transport network.
    
    Returns:
    --------
    zone_travel_times: dict
        Contains as keys tuples (node_i, node_j) and as values its travel times T_ij.
        Node indices are from original data.
    """
    zone_travel_times = {}

    # effectively make a copy of model.graph
    inds_to_nodes, graph_correspondences_, graph_table_ = model.inds_to_nodes.copy(), \
                                                            model.graph_correspondences.copy(), \
                                                            model.graph_table.copy()
    graph_dh = tg.TransportGraph(graph_table_, len(inds_to_nodes), graph_data['links number'])

    # graph_correspondences_ have its nodes reindexed
    for source, target_dict in graph_correspondences_.items():
        targets = target_dict['targets']
        travel_times, _ = graph_dh.shortest_distances(source, targets, t)
        # turn indices back to original
        source_nodes = [inds_to_nodes[source]] * len(targets)
        target_nodes = list(map(inds_to_nodes.get, targets))
        zone_travel_times.update(zip(zip(source_nodes, target_nodes), travel_times))

    return zone_travel_times


def T_matrix_from_dict(T_dict: dict, shape: tuple, old_to_new: dict) -> np.ndarray:
    """
    Create matrix from T_dict. T_dict should contain original nodes indices.
    
    Parameters:
    -----------
    T_dict: dict
        Keys - (node_i, node_j), values - T_ij
    shape: tuple of ints
        Shape of output matrix
    old_to_new: dict(int: int)
        Mapping between original nodes indices and new.
    
    Returns:
    --------
    T_matrix: np.ndarray
        Array of given shape. Indices of T_matrix and T_dict
        are linked with old_to_new mapping.
    """
    T = np.zeros(shape)
    for key in T_dict.keys():
        source, target = old_to_new[key[0]], old_to_new[key[1]]
        T[source][target] = T_dict[key]
    return T


def get_T_new(n, T, paycheck):

    T_new = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            T_new[i][j] = T[i][j] - paycheck[j]

    return T_new
