from math import sqrt
import numpy as np
from history import History


# use object to keep variables between calls in two-stage optimisation

class Ustm:
    def __init__(self, L_init, oracle, primal_dual_oracle, t_start, stop_crit, eps, eps_abs):
        self.L_value = L_init if L_init is not None else np.linalg.norm(oracle.grad(t_start))

        self.A_prev = 0.0
        self.y_start = self.u_prev = self.t_prev = np.copy(t_start)
        self.A = self.u = self.t = self.y = None

        self.grad_sum = None
        self.grad_sum_prev = np.zeros(len(t_start))

        self.flows_weighted = primal_dual_oracle.get_flows(self.y_start)
        self.primal, self.dual, self.duality_gap_init, self.state_msg = \
            primal_dual_oracle(self.flows_weighted, self.y_start)

        self.eps_abs = eps_abs
        self.eps = eps
        if self.eps_abs is None:
            self.eps_abs = eps * self.duality_gap_init

        self.success = False
        if stop_crit == 'dual_gap_rel':
            self.crit = lambda: self.duality_gap <= self.eps * self.duality_gap_init
        elif stop_crit == 'dual_gap':
            self.crit = lambda: self.duality_gap <= self.eps_abs
        elif stop_crit == 'max_iter':
            self.crit = lambda: self.it_counter == self.max_iter
        elif callable(stop_crit):
            self.crit = stop_crit
        else:
            raise ValueError("stop_crit should be callable or one of the following names: \
                             'dual_gap', 'dual_gap_rel', 'max iter'")

    def run(self, max_iter, oracle, prox, primal_dual_oracle):
        inner_iters_num = 0
        self.max_iter=max_iter
        for self.it_counter in range(1, max_iter):
            while True:
                inner_iters_num += 1

                alpha = 0.5 / self.L_value + sqrt(0.25 / self.L_value ** 2 + self.A_prev / self.L_value)
                print('L_value, A, alpha', self.L_value, self.A, alpha)
                self.A = self.A_prev + alpha
                self.y = (alpha * self.u_prev + self.A_prev * self.t_prev) / self.A
                grad_y = oracle.grad(self.y)
                self.grad_sum = self.grad_sum_prev + alpha * grad_y
                self.u = prox(self.grad_sum / self.A, self.y_start, 1.0 / self.A)
                self.t = (alpha * self.u + self.A_prev * self.t_prev) / self.A

                left_value = (oracle.func(self.y) + np.dot(grad_y, self.t - self.y) +
                              0.5 * alpha / self.A * self.eps_abs) - oracle.func(self.t)
                right_value = - 0.5 * self.L_value * np.sum((self.t - self.y) ** 2)

                if left_value >= right_value:
                    flows = primal_dual_oracle.get_flows(self.y)  # grad() is called here
                    break
                else:
                    self.L_value *= 2

            self.A_prev = self.A
            self.L_value /= 2

            self.t_prev = self.t
            self.u_prev = self.u
            self.grad_sum_prev = self.grad_sum
            print('A, alpha', self.A, alpha)
            self.flows_weighted = (self.flows_weighted * (self.A - alpha) + flows * alpha) / self.A

            self.primal, self.dual, self.duality_gap, self.state_msg = primal_dual_oracle(self.flows_weighted, self.t)

            # if save_history:
            #     history.update(it_counter, primal, dual, duality_gap, inner_iters_num)
            # if verbose and (it_counter % verbose_step == 0):
            #     print('\nIterations number: {:d}'.format(it_counter))
            #     print('Inner iterations number: {:d}'.format(inner_iters_num))
            #     print(state_msg, flush=True)
            if self.crit():
                self.success = True
                break

        result = {'times': self.t, 'flows': self.flows_weighted,
                  'iter_num': self.it_counter,
                  'res_msg': 'success' if self.success else 'iterations number exceeded'}
        # if save_history:
        #     result['history'] = history.dict
        # if verbose:
        #     print('\nResult: ' + result['res_msg'])
        #     print('Total iters: ' + str(it_counter))
        #     print(state_msg)
        #     print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))

        return result


ustm = None


def universal_similar_triangles_method(oracle, prox, primal_dual_oracle,
                                       t_start, L_init=None, max_iter=1,  # 1000,
                                       eps=1e-5, eps_abs=None, stop_crit='max_iter',
                                       verbose_step=100, verbose=False, save_history=False):
    global ustm
    if ustm is None:
        ustm = Ustm(L_init, oracle, primal_dual_oracle, t_start, stop_crit, eps, eps_abs)
    # if save_history:
    #     history = History('iter', 'primal_func', 'dual_func', 'dual_gap', 'inner_iters')
    #     history.update(0, primal, dual, duality_gap_init, 0)
    # if verbose:
    #     print(state_msg)

    return ustm.run(max_iter, oracle, prox, primal_dual_oracle)

# print('Dijkstra elapsed time: {:.0f} sec'.format(oracle.auto_oracles_time))

# criteria: stable dynamic 'dual_threshold' AND 'primal_threshold', 'dual_rel' AND 'primal_rel'.

# beckman : + 'dual_gap_rel', 'dual_gap_threshold', 'primal_threshold', 'primal_rel'

# criteria: 'star_solution_residual',

# practice: 'dual_rel'


#     if crit_name == 'dual_gap_rel':
#         def crit():
#             nonlocal duality_gap, duality_gap_init, eps
#             return duality_gap < eps * duality_gap_init
#     if crit_name == 'dual_rel':
#         def crit():
#             nonlocal dual_func_history, eps
#             l = len(dual_func_history)
#             return dual_func_history[l // 2] - dual_func_history[-1] \
#                    < eps * (dual_func_history[0] - dual_func_history[-1])
#     if crit_name == 'primal_rel':
#         def crit():
#             nonlocal primal_func_history, eps
#             l = len(primal_func_history)
#             return primal_func_history[l // 2] - primal_func_history[-1] \
#                    < eps * (primal_func_history[0] - primal_func_history[-1])
