import warnings
import sys
from pathlib import Path
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import model as md
import csv

from conf import parsers

nodes_name = None

# TODO: IDK what is best_sink_beta, maybe gamma from paper?
best_sink_beta = 0.001
sink_steps, sink_eps, sink_eps_f, sink_eps_eq = 25000, 10 ** (-8), 10 ** (-8), 10 ** (-8)
INF_COST = 100
INF_TIME = 1e10


def get_times_inverse_func(capacity, times, rho=0.15, mu=0.25):
    capacities = capacity.to_numpy()
    freeflowtimes = times
    return np.transpose((capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))


def get_LW(L_dict, W_dict, new_to_old):
    """
    Given L- and W-dicts and new nodes mapping, return L and W as np.arrays.
    Also returns people_num, total sum of L's.
    Resulting arrays contain on i'th place value for new_to_old[i] node.
    Also, L and W are divided by their means to satisfy constraints from paper.
    ----------
    Arguments:
        L_dict: dict(int: int)
            mapping between nodes and their l's
        W_dict: dict(int: int)
            mapping between nodes and their w's
        new_to_old: dict(int: int)
            mapping between old indices of nodes and new
    ----------
    Returns:
        L: np.array
            Normalized array of l values.
        w: np.array
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

def create_results_dir():
    """
    Create folders necessary for storing results.
    There is no arguments because it would be a pain the ass.
    """
    results_path = Path('results/')
    children = []
    for child in ['input_data', 'iter', 'multi']:
        children.append(results_path / child)
    for child in ['corr_matrix', 'flows', 'inverse_func', 'subg', 'times']:
        children.append(children[2] / child)
    [child.mkdir(exist_ok=True) for child in children]


if __name__ == '__main__':
    create_results_dir()
    # read data from txt files
    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(eval(f'handler.{parsers}_net_parser'),
                                      columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])

    L_dict, W_dict = handler.GetLW_dicts(eval(f'handler.{parsers}_corr_parser'))
    handler.save_input_data_to_res(graph_data, L_dict, W_dict)

    # don't understand why DataHandler is reinitialized.
    handler = dh.DataHandler()

    max_iter = 2
    alpha = 0.9

    empty_corr_dict = {source: {'targets': list(W_dict.keys())} for source in L_dict.keys()}
    empty_corr_matrix, old_to_new, new_to_old = handler.reindexed_empty_corr_matrix(empty_corr_dict)
    print('fill correspondence_matrix')

    L, W, people_num = get_LW(L_dict, W_dict, new_to_old)
    total_od_flow = people_num

    print(f'L, W, people_num {L, W, people_num}, total_od_flow: {total_od_flow}')
    model = md.Model(graph_data, empty_corr_dict, total_od_flow, mu=0.25)
    # initialize dict with T_ij values from free flow times as first iteration
    T_dict = handler.get_T_from_t(graph_data['graph_table']['free_flow_time'],
                                  graph_data, model)
    T = handler.T_matrix_from_dict(T_dict, empty_corr_matrix.shape, old_to_new)
    T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)
    T = np.nan_to_num(T * best_sink_beta, nan=INF_COST)

    for ms_i in range(12):

        print('iteration: ', ms_i)

        algorithm = sys.argv[1]
        if algorithm == 'base':
            s = skh.Sinkhorn(L, W, people_num, sink_steps, sink_eps)
            cost_matrix = T
            print('cost matrix', cost_matrix)
            d_hat, _, _ = s.iterate(cost_matrix)
        elif algorithm == 'accelerated':
            s = skh.AcceleratedSinkhorn(L, W, T, people_num, 
                                        sink_steps, sink_eps_f, sink_eps_eq)
            d_hat, x = s.iterate()
        print('rec', d_hat, np.sum(d_hat))
        sink_correspondences_dict = handler.corr_matrix_to_dict(d_hat, new_to_old)

        L_new = np.nansum(d_hat, axis=1)
        L_new /= np.nansum(L_new)
        W_new = np.nansum(d_hat, axis=0)
        W_new /= np.nansum(W_new)

        model.refresh_correspondences(graph_data, sink_correspondences_dict)

        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])

        model.graph.update_flow_times(result['times'])

        print(result.keys(), np.shape(result['flows']))
        for flow, time in zip(result['flows'], result['times']):
            print('flow, time: ', flow, time)

        T_dict = result['zone travel times']
        T = handler.T_matrix_from_dict(T_dict, d_hat.shape, old_to_new)
        flows_inverse_func = get_times_inverse_func(graph_data['graph_table']['capacity'], result['times'], rho=0.15,
                                                    mu=0.25)

        subg = result['subg']

        np.savetxt('results/multi/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
        np.savetxt('results/multi/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
        np.savetxt('results/multi/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', d_hat, delimiter=' ')
        np.savetxt('results/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                    delimiter=' ')
        np.savetxt('results/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')