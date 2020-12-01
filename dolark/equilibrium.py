import scipy
from dolo import colored
import numpy as np
import pandas as pd
from .shocks import inject_process
from dolo import improved_time_iteration, time_iteration, ergodic_distribution

from .shocks import discretize_idiosyncratic_shocks

class Equilibrium:

    def __init__(self, aggmodel, m, μ, dr, y):
        self.m = m
        self.μ = μ
        self.dr = dr
        self.x = np.concatenate([e[None,:,:] for e in [dr(i,dr.endo_grid.nodes) for i in range(max(dr.exo_grid.n_nodes,1))] ], axis=0)
        self.y = y
        self.c = dr.coefficients

        self.states = np.concatenate([e.ravel() for e in (m, μ)])
        self.controls = np.concatenate([e.ravel() for e in (self.x, y)])
        self.aggmodel = aggmodel

    def as_df(self):
        model = self.aggmodel.model
        eq = self
        exg = np.column_stack([range(eq.dr.exo_grid.n_nodes), eq.dr.exo_grid.nodes])
        edg = np.column_stack([eq.dr.endo_grid.nodes])
        N_m = exg.shape[0]
        N_s = edg.shape[0]
        ssg = np.concatenate([exg[:,None,:].repeat(N_s, axis=1), edg[None,:,:].repeat(N_m, axis=0)], axis=2).reshape((N_m*N_s,-1))
        x = np.concatenate([eq.dr(i, edg) for i in range(max(eq.dr.exo_grid.n_nodes,1))], axis=0)
        import pandas as pd
        cols = ['i_m'] + model.symbols['exogenous'] + model.symbols['states'] + ['μ'] + model.symbols['controls']
        df = pd.DataFrame(np.column_stack([ssg, eq.μ.ravel(), x]), columns=cols)
        return df


def equilibrium(hmodel, m0: 'vector', y0: 'vector', p=None, dr0=None, grids=None, verbose=False, return_equilibrium=True):
    if p is None:
        p = hmodel.calibration['parameters']

    q0 = hmodel.projection(m0, y0, p)

    dp = inject_process(q0, hmodel.model.exogenous)

    sol = improved_time_iteration(hmodel.model, dr0=dr0, dprocess=dp, verbose=verbose)
    dr = sol.dr

    if grids is None:
        exg, edg = grids = dr.exo_grid, dr.endo_grid
    else:
        exg, edg = grids

    Π0, μ0 = ergodic_distribution(hmodel.model, dr, exg, edg, dp)

    s = edg.nodes
    if exg.n_nodes==0:
        nn = 1
        μμ0 = μ0.data[None,:]
    else:
        nn = exg.n_nodes
        μμ0 = μ0.data

    xx0 = np.concatenate([e[None,:,:] for e in [dr(i,s) for i in range(nn)] ], axis=0)

    res = hmodel.𝒜(grids, m0, μμ0, xx0, y0, p)

    if return_equilibrium:
        return (res, sol, μ0, Π0)
    else:
        return res


def find_steady_state(hmodel, dr0=None, verbose=True, distribs=None):

    m0 = hmodel.calibration['exogenous']
    y0 = hmodel.calibration['aggregate']
    p = hmodel.calibration['parameters']

    if dr0 is None:
        if verbose: print("Computing Initial Initial Rule... ", end="")
        dr0 = hmodel.get_starting_rule()
        if verbose: print(colored("done", "green"))

    if verbose: print("Computing Steady State...", end="")

    if distribs is None:
        dist = [(1.0, {})]
        if not hmodel.features['ex-ante-identical']:
            dist = distribs = discretize_idiosyncratic_shocks(hmodel.distribution)
    else:
        dist = distribs

    def fun(u):
        res = y0*0
        for w, kwargs in dist:
            hmodel.model.set_calibration(**kwargs)
            res += w*equilibrium(hmodel,
                            m0,
                            u,
                            dr0=dr0,
                            return_equilibrium=False)
        return res

    solution = scipy.optimize.root(fun, x0=y0)
    if not solution.success:
        if verbose: print(colored("failed", "red"))
    else:
        if verbose: print(colored("done", "green"))


    # grid_m = model.exogenous.discretize(to='mc', options=[{},{'N':N_mc}]).nodes
    # grid_s = model.get_grid().nodes
    #
    y_ss = solution.x # vector of aggregate endogenous variables
    m_ss = m0 # vector fo aggregate exogenous
    eqs = []
    for w, kwargs in (dist):
        hmodel.model.set_calibration(**kwargs)
        (res_ss, sol_ss, μ_ss, Π_ss) = equilibrium(hmodel, m_ss, y_ss, p, dr0, return_equilibrium=True)
        μ_ss = μ_ss.data
        dr_ss = sol_ss.dr
        eqs.append([w, Equilibrium(hmodel, m_ss, μ_ss, sol_ss.dr, y_ss)])

    if distribs is None:
        return eqs[0][1]
    else:
        return eqs
