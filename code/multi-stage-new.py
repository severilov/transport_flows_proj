import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import model as md
import csv

from conf import parsers

nodes_name = None

best_sink_beta = 0.001
sink_num_iter, sink_eps = 25000, 10 ** (-8)
INF_COST = 100
INF_TIME = 1e10


def get_times_inverse_func(capacity, times, rho=0.15, mu=0.25):
    capacities = capacity.to_numpy()
    freeflowtimes = times
    return np.transpose((capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))


def get_LW(L_dict, W_dict, new_to_old):
    L = np.array([L_dict[new_to_old[i]] for i in range(len(L_dict))], dtype=np.double)
    W = np.array([W_dict[new_to_old[i]] for i in range(len(W_dict))], dtype=np.double)
    people_num = L.sum()
    L /= np.nansum(L)
    W /= np.nansum(W)
    return L, W, people_num


if __name__ == '__main__':

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(eval(f'handler.{parsers}_net_parser'),
                                      columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])

    L_dict, W_dict = handler.GetLW_dicts(eval(f'handler.{parsers}_corr_parser'))
    handler.save_input_data_to_res(graph_data, L_dict, W_dict)

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
    T_dict = handler.get_T_from_t(graph_data['graph_table']['free_flow_time'],
                                  graph_data, model)
    T = handler.T_matrix_from_dict(T_dict, empty_corr_matrix.shape, old_to_new)

    for ms_i in range(12):

        print('iteration: ', ms_i)

        s = skh.Sinkhorn(L, W, people_num, sink_num_iter, sink_eps)
        T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)

        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=INF_COST)

        print('cost matrix', cost_matrix)
        rec, _, _ = s.iterate(cost_matrix)
        print('rec', rec, np.sum(rec))
        sink_correspondences_dict = handler.corr_matrix_to_dict(rec, new_to_old)

        L_new = np.nansum(rec, axis=1)
        L_new /= np.nansum(L_new)
        W_new = np.nansum(rec, axis=0)
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
        T = handler.T_matrix_from_dict(T_dict, rec.shape, old_to_new)
        flows_inverse_func = get_times_inverse_func(graph_data['graph_table']['capacity'], result['times'], rho=0.15,
                                                    mu=0.25)

        subg = result['subg']

        np.savetxt('KEV_res/multi/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
        np.savetxt('KEV_res/multi/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
        np.savetxt('KEV_res/multi/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', rec, delimiter=' ')
        np.savetxt('KEV_res/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                    delimiter=' ')
        np.savetxt('KEV_res/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')