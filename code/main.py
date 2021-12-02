import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import tensorflow as tf
import tensorflow.distributions
# from tensorflow.distributions import Dirichlet, Multinomial
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import oracles as oracle
import model as md
import time
import pickle
import transport_graph as tg

if __name__ == '__main__':


    net_name = 'Custom_net.tntp'
    trips_name = 'Custom_trips.tntp'

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, columns = ['init_node', 'term_node', 'capacity', 'free_flow_time'])
    graph_correspondences, total_od_flow = handler.GetLW_dicts(trips_name)

    handler = dh.DataHandler()

    model = md.Model(graph_data, graph_correspondences,
                     total_od_flow, mu=0)

    max_iter = 2000
    alpha = 0.9

    graph_table = graph_data['graph_table']

    # print(model.graph.links_number, model.graph.nodes_number, model.graph)
    # print(graph_data['links number'], graph_data['nodes number'])
    # # print('model: ', len(self.inds_to_nodes), graph_data['links number'])
    #
    # graph = tg.TransportGraph(graph_table,
    #                           graph_data['links number'],
    #                           graph_data['nodes number'])


    for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
        print(i, eps_abs)
        solver_kwargs = {'eps_abs': eps_abs,
                         'max_iter': max_iter}

        result = model.find_equilibrium(solver_name='ustm', composite=True,
                                        solver_kwargs=solver_kwargs,
                                        base_flows=alpha * graph_data['graph_table']['capacity'])

        print(result.keys())
        print('eps_abs =', eps_abs)
        print('flows: ',  result['flows'])
        print('times: ', result['times'])
        print('zone travel times', result['zone travel times'])