# model parameters:
import copy
import numpy as np
import pandas as pd
import transport_graph as tg

import oracles
import dual_func_calculator as dfc

import solvers.universal_similar_triangles_method as ustm
import solvers.universal_gradient_descent_method as ugd
import solvers.subgradient_descent_method as sd
import solvers.frank_wolfe_method as fwm
import solvers.weighted_dual_averages_method as wda


class Model:
    def __init__(self, graph_data: pd.DataFrame, graph_correspondences: dict, total_od_flow, mu=0.25, rho=0.15):
        """
        Reindex nodes according to their roles in graph and create TransportGraph instance.
        
        Parameters:
        -----------
        graph_data: pd.Dataframe
            This is output dict of DataHandler.getGraphData.
        graph_correspondencies: dict
            This is dict of form like 'empty_corr_dict' from multi-stage-new.py
        total_od_flow: int
            Total number of people or, formally, sum of l's from paper.
        mu: float
            Parameter for computing cost function tau(f) from paper.
        rho: float
            Хз что это такое, нужно в Оракулах.
        """
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho
        self.inds_to_nodes, self.graph_correspondences, self.graph_table = \
            self._index_nodes(graph_data['graph_table'], graph_correspondences, fill_corrs=False)
        self.graph = tg.TransportGraph(self.graph_table, len(self.inds_to_nodes), graph_data['links number'])

    def refresh_correspondences(self, graph_data: pd.DataFrame, corrs_dict: dict) -> None:
        """
        Wrapper for Model._index_nodes method. See docstring for that method for more info.
        
        Parameters:
        -----------
        graph_table: pd.Dataframe
            This is 'graph_table' from `data-handler.getGraphData` output dict.
        graph_correspondencies: dict
            This is dict of form like `empty_corr_dict` from `multi-stage-new.py`

        Returns:
        --------
        None
        """
        self.inds_to_nodes, self.graph_correspondences, _ = self._index_nodes(graph_data['graph_table'], corrs_dict)

    @staticmethod
    def _index_nodes(graph_table: pd.DataFrame, graph_correspondences: dict, fill_corrs=True) -> tuple:
        """
        Another reindexing of nodes. This time they are indexed to ensure
        that 'true' init nodes are first, then 'through' nodes and then 'true' term.
        
        Parameters:
        ----------
        graph_table: pd.Dataframe
            This is 'graph_table' from DataHandler.getGraphData output dict.
        graph_correspondencies: dict
            This is dict of form like 'empty_corr_dict' from multi-stage-new.py
        fill_corrs: bool
            If graph_correspondencies dict already have correspondence values,
            fill'em in reindexed dict.
        
        Returns:
        --------
        inds_to_nodes: dict(int: int)
            Mapping from new nodes indices to old.
        correspondences: dict
            Reindexed graph_correspondences.
        table: pd.Dataframe
            graph_table with reindexed 'init_node' and 'term_node' columns.
        """
        table = graph_table.copy()
        # Don't completely understand what happens here either.
        # These inits and terms are those with id's less than 
        # 'first_thru_node' parameter of data-handler.vladik_net_parser
        # I suppose these are 'true' inits and terms, and through_nodes are just intermediate?
        inits = np.unique(table['init_node'][table['init_node_thru'] == False])
        terms = np.unique(table['term_node'][table['term_node_thru'] == False])
        through_nodes = np.unique([table['init_node'][table['init_node_thru'] == True],
                                   table['term_node'][table['term_node_thru'] == True]])
        nodes = np.concatenate((inits, through_nodes, terms))

        # remap nodes once again
        nodes_inds = list(zip(nodes, np.arange(len(nodes))))
        inds_to_nodes = dict(zip(range(len(nodes)), nodes))
        init_to_ind = dict(nodes_inds[: len(inits) + len(through_nodes)])
        term_to_ind = dict(nodes_inds[len(inits):])
        table['init_node'] = table['init_node'].map(init_to_ind)
        table['term_node'] = table['term_node'].map(term_to_ind)

        # get correspondeces in form of dict: origin -> dict{'targets': list of targets}
        correspondences = {}
        for origin, dests in graph_correspondences.items():
            dests = copy.deepcopy(dests)
            d = {'targets': list(map(term_to_ind.get, dests['targets']))}
            if fill_corrs:
                d['corrs'] = dests['corrs']
            correspondences[init_to_ind[origin]] = d

        return inds_to_nodes, correspondences, table

    def find_equilibrium(self, solver_name: str='ustm', composite: bool=True, 
                         solver_kwargs: dict={}, base_flows=None) -> dict:
        """
        Solve task `14` from original paper using one of available methods.

        Parameters:
        -----------
        solver_name: {'fwm', 'ustm', 'ugd', 'wda', 'sd'}, str
        composite: bool
        solver_kwargs: dict
        base_flows: pd.Series or np.ndarray

        Returns:
        --------
        results: dict
            Contains following fields:
            `'times'`: `t` from paper;
            `'zone travel times'`: `T` from paper in form of dict;
            `'subg'`: subgradient, it seems;
            other fields from optimizer.
        """
        if solver_name == 'fwm':
            solver_func = fwm.frank_wolfe_method
            starting_msg = 'Frank-Wolfe method...'
        elif solver_name == 'ustm':
            solver_func = ustm.universal_similar_triangles_method
            starting_msg = 'Universal similar triangles method...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = self.graph.max_path_length ** 0.5 * self.total_od_flow

        elif solver_name == 'ugd':
            solver_func = ugd.universal_gradient_descent_method
            starting_msg = 'Universal gradient descent method...'
            if not 'L_init' in solver_kwargs:
                solver_kwargs['L_init'] = self.graph.max_path_length ** 0.5 * self.total_od_flow

        elif solver_name == 'wda':
            solver_func = wda.weighted_dual_averages_method
            starting_msg = 'Weighted dual averages method...'

        elif solver_name == 'sd':
            solver_func = sd.subgradient_descent_method
            starting_msg = 'Subgradient descent method...'

        else:
            raise NotImplementedError('Unknown solver!')

        phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)

        h_oracle = oracles.HOracle(self.graph.initial_times, self.graph.capacities,
                                   rho=self.rho, mu=self.mu)
        primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle, h_oracle,
                                                          self.graph.initial_times, self.graph.capacities,
                                                          rho=self.rho, mu=self.mu, base_flows=base_flows)
        if composite == True or solver_name == 'fwm':
            if not solver_name == 'fwm':
                print('Composite optimization...')
            oracle = phi_big_oracle
            prox = h_oracle.prox
        else:
            print('Non-composite optimization...')
            oracle = phi_big_oracle + h_oracle

            def prox_func(grad, point, A):
                """
                Computes argmin_{t: t \in Q} <g, t> + A / 2 * ||t - p||^2
                    where Q - the feasible set {t: t >= free_flow_times},
                    A - constant, g - (sub)gradient vector, p - point at which prox is calculated
                """
                return np.maximum(point - grad / A, self.graph.initial_times)

            prox = prox_func
        print('Oracles created...')
        print(starting_msg)

        if solver_name == 'fwm':
            result = solver_func(oracle,
                                 primal_dual_calculator,
                                 t_start=self.graph.initial_times,
                                 **solver_kwargs)
        else:
            result = solver_func(oracle, prox,
                                 primal_dual_calculator,
                                 t_start=self.graph.initial_times,
                                 **solver_kwargs)
        # getting travel times of every non-zero trips between zones:
        result['zone travel times'] = {}
        result['subg'] = {}
        subg_t = phi_big_oracle.grad(result['times'])
        # print('att! ', subg_t, np.shape(subg_t))
        result['subg'] = subg_t

        for source, targets_dict in self.graph_correspondences.items():
            targets = targets_dict['targets']
            travel_times, pred_map = self.graph.shortest_distances(source, targets, result['times'])

            # subg_t = phi_big_oracle.grad(result['times'])

            # print('types: ', type(subg_t), type(travel_times))
            # print('in model.py, travel_times: ', travel_times)

            # mapping nodes' indices to initial nodes' names:
            source_nodes = [self.inds_to_nodes[source]] * len(targets)
            target_nodes = list(map(self.inds_to_nodes.get, targets))

            result['zone travel times'].update(zip(zip(source_nodes, target_nodes), travel_times))

        return result
