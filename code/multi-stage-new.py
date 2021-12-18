import warnings
import sys
from pathlib import Path
import os
import time
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

np.set_printoptions(suppress=True)

import sinkhorn as skh
import data_handler as dh
import model as md

from conf import net_name


nodes_name = None

best_sink_beta = 10 ** (-3)
sink_steps, sink_eps = 25000, 10 ** (-8)
sink_eps_f, sink_eps_eq = 10 ** (-2), 10 ** (-2)
max_iter = 2
alpha = 0.9
rho = 0.15
mu = 0.25
INF_COST = 100
INF_TIME = 1e10


def flows_from_times(capacity: pd.Series, times, freeflowtimes: pd.Series=None, rho=0.15, mu=0.25) -> np.ndarray:
    """
    Inverse function of formula `2` from original paper `https://arxiv.org/pdf/2012.04516.pdf`.
    E.g. we find flow from time.

    Parameters:
    -----------
    capacity: pd.Series
        Пропускная способность ребер (f_e с чертой из статьи).
    times: 
        \Tau из статьи, затраты на проезд по ребрам.
    rho: float
        `k` from formula `2`.
    mu: float
        `mu` from formula `2`.

    Returns:
    --------
    flow: np.ndarray
        `f_e` from formula `2`.
    """
    capacities = capacity.to_numpy()
    if freeflowtimes is None:
        freeflowtimes = times # this is by default and it's strange   
    return np.transpose((capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))


def create_results_dir() -> None:
    """
    Create folders necessary for storing results.
    There is no parameters because it would be a pain in the ass.
    """
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    results_path = Path('results/')
    results_path.mkdir(exist_ok=True)
    children = []
    for child in ['input_data', 'iter', 'multi']:
        children.append(results_path / child)
    for child in ['corr_matrix', 'flows', 'inverse_func', 'subg', 'times']:
        children.append(children[2] / child)
    [child.mkdir(exist_ok=True) for child in children]


if __name__ == '__main__':
    create_results_dir()
    # read data from txt files
    graph_data, L_dict, W_dict = dh.parse_data(net_name)
    dh.save_input_data_to_res(graph_data, L_dict, W_dict)

    # create initial correspondence matrix, L-dict and W-dict in fact have same keys
    empty_corr_dict = {source: {'targets': list(W_dict.keys())} for source in L_dict.keys()}
    empty_corr_matrix, old_to_new, new_to_old = dh.reindexed_empty_corr_matrix(empty_corr_dict)
    #print('fill correspondence_matrix')

    L, W, people_num = dh.get_LW(L_dict, W_dict, new_to_old) # all values in L, W should be in range [0, 1]
    total_od_flow = people_num
    #print(f'L, W, people_num {L, W, people_num}, total_od_flow: {total_od_flow}')
    
    model = md.Model(graph_data, empty_corr_dict, total_od_flow, mu=mu)

    # initialize dict with T_ij values from `free_flow_time` as first iteration
    T_dict = dh.get_T_from_t(graph_data['graph_table']['free_flow_time'],
                                  graph_data, model)
    T = dh.T_matrix_from_dict(T_dict, empty_corr_matrix.shape, old_to_new)
    T = np.nan_to_num(T * best_sink_beta, nan=INF_COST, posinf=INF_COST, neginf=INF_COST)
    T[T == 0.0] = 100.0

    for ms_i in range(22):

        print('-'*20 + f'iteration: {ms_i}' + '-'*20)

        # find reconstructed correspondence matrix d_hat
        # e.g find new d(T)
        algorithm = sys.argv[1]
        if algorithm == 'base':
            s = skh.Sinkhorn(L, W, people_num, sink_steps, sink_eps)
            cost_matrix = T
            d_hat, _, _ = s.iterate(cost_matrix)
        elif algorithm == 'accelerated':
            s = skh.AcceleratedSinkhorn(L, W, T, people_num, 
                                        sink_steps, sink_eps_f, sink_eps_eq)
            d_hat, x = s.iterate()
        #print('rec', d_hat, np.sum(d_hat))
        sink_correspondences_dict = dh.corr_matrix_to_dict(d_hat, new_to_old)

        # synchronize indices of correspondence matrix
        model.refresh_correspondences(graph_data, sink_correspondences_dict)

        # solve task `14` from original paper `https://arxiv.org/pdf/2012.04516.pdf`.
        # e.g. find new T(d)
        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])
        model.graph.update_flow_times(result['times'])

        # update T
        T_dict = result['zone travel times']
        T = dh.T_matrix_from_dict(T_dict, d_hat.shape, old_to_new)

        # calculate flows from times
        flows = flows_from_times(graph_data['graph_table']['capacity'],
                                 graph_data['graph_table']['free_flow_time'],
                                 result['times'], rho=rho, mu=mu)

        subg = result['subg']

        np.savetxt(f'results/multi/flows/{algorithm}_' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
        np.savetxt(f'results/multi/times/{algorithm}_' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
        np.savetxt(f'results/multi/corr_matrix/{algorithm}_' + str(ms_i) + '_corr_matrix.txt', d_hat, delimiter=' ')
        np.savetxt(f'results/multi/inverse_func/{algorithm}_' + str(ms_i) + '_inverse_func.txt', flows, delimiter=' ')
        np.savetxt(f'results/multi/subg/{algorithm}_' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')
