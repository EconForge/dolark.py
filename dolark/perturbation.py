from dolo import colored
from .dolo_improvements import *
import scipy.optimize

from matplotlib import pyplot as plt
from dolo.algos.ergodic import ergodic_distribution
from numpy import exp
import copy

# from bl_functions import *
from dolo import *
import numpy as np
import copy
import tqdm

# import dolo_improvements
from .dolo_improvements import TrickyMarkovChain



class AggregateModel:


    def __init__(self, model, dr0=None, discretization_options={}, exo_grid=None, endo_grid=None):


        if len(discretization_options) == 0:
            discretization_options = {"N": 2}


        if exo_grid is not None:
            raise Exception("Not implemented.")
        if endo_grid is not None:
            raise Exception("Not implemented.")

        self.__variables__ = None
        self.__project_indices__ = None

        # by default grids for the distribution are taken from model/d.r.
        exogenous = copy.deepcopy(model.exogenous)
        mc = exogenous.discretize(to='mc', options=[{}, discretization_options])
        exo_grid = mc.grid
        endo_grid = model.get_grid()
        self.grids = (exo_grid, endo_grid)

        if dr0 is None:
            sol0 = improved_time_iteration(model, details=True, dp=mc)
            # TODO: check whether solution is good
            dr0 = sol0.dr

        self.model = model
        self.dr0 = dr0
        self.discretization_options = discretization_options


    def equilibrium(self, m0: 'vector', y0: 'vector', p, dr0, verbose=False):

        q0 = [*self.projection(m0, y0, p).values()]

        v0 = np.concatenate([m0, y0, q0])

        exogenous = copy.deepcopy(self.model.exogenous)
        exogenous.processes[0].μ = np.array(q0)

        mc = exogenous.discretize(to='mc', options=[{},self.discretization_options])

        sol = improved_time_iteration(self.model, initial_dr=dr0, dp=mc, verbose=verbose)
        dr = sol.dr

        exg, edg = self.grids

        Π0, μ0 = ergodic_distribution(self.model, dr, exg, edg, mc)
        s = dr.endo_grid.nodes()
        xx0 = np.concatenate([e[None,:,:] for e in [dr(i,s) for i in range(mc.n_nodes())] ], axis=0)

        res = self.𝒜(m0, μ0, xx0, y0, p)
        return res, (sol, μ0, Π0)

    def get_starting_rule(self):
        # provides initial guess for d.r. by solving agent's problem

        mc = self.model.exogenous.discretize(to='mc', options=[{},self.discretization_options])
        # dr0 = time_iteration(self.model, dprocess=mc)
        sol = improved_time_iteration(self.model, dp=mc)
        dr0 = sol.dr

        return dr0

    def find_steady_state(self, dr0=None, verbose=True):

        m0 = self.calibration['exogenous']
        y0 = self.calibration['aggregate']
        p = self.calibration['parameters']

        if dr0 is None:
            if verbose: print("Computing Initial Initial Rule... ", end="")
            dr0 = self.get_starting_rule()
            if verbose: print(colored("done", "green"))

        if verbose: print("Computing Steady State...", end="")
        fun = lambda u: self.equilibrium(m0, u, p, dr0)[0]


        solution = scipy.optimize.root(fun, x0=y0)
        if not solution.success:
            if verbose: print(colored("failed", "red"))

        if verbose: print(colored("done", "green"))


        # grid_m = model.exogenous.discretize(to='mc', options=[{},{'N':N_mc}]).nodes()
        # grid_s = model.get_grid().nodes()
        #
        y_ss = solution.x # vector of aggregate endogenous variables
        m_ss = m0 # vector fo aggregate exogenous
        (sol_ss, μ_ss, Π_ss) = self.equilibrium(m_ss, y_ss, p, dr0)[1]
        μ_ss = μ_ss.data
        dr_ss = sol_ss.dr

        return Equilibrium(self, m_ss, μ_ss, sol_ss.dr, y_ss)

    def G(self, equilibrium, states_p, controls_p, p):

        eq = equilibrium
        m_ss = eq.m
        μ_ss = eq.μ
        x_ss = eq.x
        y_ss = eq.y


        m0, μ0 = unpack(states_p, (m_ss, μ_ss))
        x0, y0 = unpack(controls_p, (x_ss, y_ss))


        q0 = np.array([*self.projection(m0, y0, p).values()])
        exogenous = copy.deepcopy(self.model.exogenous)
        exogenous.processes[0].μ = q0
        mc = exogenous.discretize(to='mc', options=[{},self.discretization_options])
        # that should actually depend on m1

        dr0 = copy.deepcopy(eq.dr)
        dr0.set_values(x0)


        exg, edg = self.grids
        # we should not compute μ here...
        μ0 = μ0.ravel()
        Π0,_ = ergodic_distribution(self.model, dr0, exg, edg, mc)

        k = len(μ0)
        Π0 = Π0.reshape((k,k))

        m1 = self.τ(m0, p)
        μ1 = μ0@Π0

        return pack([m1, μ1])


    def F(self, equilibrium, states, controls, states_f, controls_f, p):

        eq = equilibrium
        m_ss = eq.m
        μ_ss = eq.μ
        x_ss = eq.x
        y_ss = eq.y

        m0, μ0 = unpack(states, (m_ss, μ_ss))
        x0, y0 = unpack(controls, (x_ss, y_ss))
        m1, μ1 = unpack(states_f, (m_ss, μ_ss))
        x1, y1 = unpack(controls_f, (x_ss, y_ss))


        q0 = [*self.projection(m0, y0, p).values()]
        _m0 = np.array(q0)

        q1 = [*self.projection(m1, y1, p).values()]
        _m1 = np.array(q1)


        exogenous = copy.deepcopy(self.model.exogenous)
        _mc = exogenous.processes[1].discretize(to='mc', **self.discretization_options)
        tmc = TrickyMarkovChain(_m0, _m1, _mc)


        dr1 = copy.deepcopy(eq.dr)
        dr1.set_values(x1)

        dr0 = time_iteration(self.model, initial_guess=dr1, verbose=False, maxit=1, dprocess=tmc)
        s = dr0.endo_grid.nodes()
        n_m = _mc.n_nodes()
        xx0 = np.concatenate([e[None,:,:] for e in [dr0(i,s) for i in range(n_m)] ], axis=0)

        res_0 = xx0-x0

        aggres_0 = self.𝒜(m0, μ0, xx0, y0, p)

        return pack([res_0, aggres_0])


    def get_derivatives(self, eq):

        p = self.calibration['parameters']
        states_ss = eq.states
        controls_ss = eq.controls

        g = self.G
        f = self.F

        g_s = jacobian(lambda u: g(eq, u, controls_ss, p), states_ss)
        g_x = jacobian(lambda u: g(eq, states_ss, u, p), controls_ss)
        g_e = np.zeros((g_s.shape[0], 1))
        f_s = jacobian(lambda u: f(eq, u, controls_ss, states_ss, controls_ss, p), states_ss)
        f_x = jacobian(lambda u: f(eq, states_ss, u, states_ss, controls_ss, p), controls_ss)
        f_S = jacobian(lambda u: f(eq, states_ss, controls_ss, u, controls_ss, p), states_ss)
        f_X = jacobian(lambda u: f(eq, states_ss, controls_ss, states_ss, u, p), controls_ss)

        return g_s, g_x, g_e, f_s, f_x, f_S, f_X

    def perturb(self, eq, verbose=True):

        from dolo.algos.perturbation import approximate_1st_order

        if verbose: print("Computing Jacobian...", end="")
        g_s, g_x, g_e, f_s, f_x, f_S, f_X = self.get_derivatives(eq)
        if verbose: print(colored("done", "green"))
        if verbose: print("Solving Perturbation...", end="")
        C0, evs = approximate_1st_order(g_s, g_x, g_e, f_s, f_x, f_S, f_X)
        if verbose: print(colored("done", "green"))
        C = C0
        P = g_s + g_x@C0

        return PerturbedEquilibrium(eq, C, P, evs)

class Equilibrium:

    def __init__(self, aggmodel, m, μ, dr, y):
        self.m = m
        self.μ = μ
        self.dr = dr
        self.x = np.concatenate([e[None,:,:] for e in [dr(i,dr.endo_grid.nodes()) for i in range(dr.exo_grid.n_nodes())] ], axis=0)
        self.y = y
        self.c = dr.coefficients

        self.states = np.concatenate([e.ravel() for e in (m, μ)])
        self.controls = np.concatenate([e.ravel() for e in (self.x, y)])
        self.aggmodel = aggmodel

    def as_df(self):
        model = self.aggmodel.model
        eq = self
        exg = np.column_stack([range(eq.dr.exo_grid.n_nodes()), eq.dr.exo_grid.nodes()])
        edg = np.column_stack([eq.dr.endo_grid.nodes()])
        N_m = exg.shape[0]
        N_s = edg.shape[0]
        ssg = np.concatenate([exg[:,None,:].repeat(N_s, axis=1), edg[None,:,:].repeat(N_m, axis=0)], axis=2).reshape((N_m*N_s,-1))
        x = np.concatenate([eq.dr(i, edg) for i in range(eq.dr.exo_grid.n_nodes())], axis=0)
        import pandas as pd
        cols = ['i_m'] + model.symbols['exogenous'] + model.symbols['states'] + ['μ'] + model.symbols['controls']
        df = pd.DataFrame(np.column_stack([ssg, eq.μ.ravel(), x]), columns=cols)
        return df

class PerturbedEquilibrium:

    def __init__(self, eq, C, P, evs):
        self.eq = eq
        self.C = C
        self.P = P
        self.evs = evs

    def response(peq, m0, T=200):

        eq = peq.eq

        m0 = np.array(m0)
        C = peq.C
        P = peq.P
        n_s = eq.states.shape[0]
        n_x = eq.controls.shape[0]
        svec = np.zeros((T+1, n_s))
        xvec = np.zeros((T+1, n_x))
        s0 = pack([m0, eq.μ*0])
        svec[0,:] = s0
        for t in range(T):
            svec[t+1,:] = P@svec[t,:]
        for t in range(T+1):
            xvec[t,:] = C@svec[t,:]
        svec[:,:] += eq.states[None,:]
        xvec[:,:] += eq.controls[None,:]
        # not clear what object to return here:
        vec = np.concatenate([svec, xvec], axis=1)
        return [unpack(vec[t,:], [eq.m, eq.μ, eq.x, eq.y]) for t in range(T+1)]

    def simul(self, dz):

        C = self.C
        P = self.P
        m_ss = self.eq.m
        μ_ss = self.eq.μ
        x_ss = self.eq.x
        y_ss = self.eq.y

        states_ss = self.eq.states
        controls_ss = self.eq.controls

        sim = []
        λ = dz
        St = states_ss.copy()
        St[0] += λ

        for i in range(100):

            Xt = controls_ss + C@(St-states_ss)

            mt, μt = unpack(St, (m_ss, μ_ss))
            xt, yt = unpack(Xt, (x_ss, y_ss))

            St = states_ss + P@(St-states_ss)

            sim.append((mt, μt, xt, yt))

        return sim

#%%
