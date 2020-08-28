from dolo import colored
from .dolo_improvements import *
import scipy.optimize

from matplotlib import pyplot as plt
from dolo.algos import ergodic_distribution, time_iteration, improved_time_iteration
from numpy import exp
import copy

# from bl_functions import *
from dolo import *
import numpy as np
import copy
import tqdm

# import dolo_improvements
from .dolo_improvements import TrickyMarkovChain


def G(hmodel, equilibrium, states_p, controls_p, p):

    eq = equilibrium
    m_ss = eq.m
    μ_ss = eq.μ
    x_ss = eq.x
    y_ss = eq.y


    m0, μ0 = unpack(states_p, (m_ss, μ_ss))
    x0, y0 = unpack(controls_p, (x_ss, y_ss))


    q0 = hmodel.projection(m0, y0, p)
    exogenous = copy.deepcopy(hmodel.model.exogenous)
    exogenous.processes[0].μ = q0
    mc = exogenous.discretize(to='mc', options=[{},hmodel.discretization_options])
    # that should actually depend on m1

    dr0 = copy.deepcopy(eq.dr)
    dr0.set_values(x0)


    exg, edg = dr0.exo_grid, dr0.endo_grid
    # we should not compute μ here...
    μ0 = μ0.ravel()
    Π0,_ = ergodic_distribution(hmodel.model, dr0, exg, edg, mc)

    k = len(μ0)
    Π0 = Π0.reshape((k,k))

    m1 = hmodel.τ(m0, p)
    μ1 = μ0@Π0

    return pack([m1, μ1])


def F(hmodel, equilibrium, states, controls, states_f, controls_f, p):

    eq = equilibrium
    m_ss = eq.m
    μ_ss = eq.μ
    x_ss = eq.x
    y_ss = eq.y

    m0, μ0 = unpack(states, (m_ss, μ_ss))
    x0, y0 = unpack(controls, (x_ss, y_ss))
    m1, μ1 = unpack(states_f, (m_ss, μ_ss))
    x1, y1 = unpack(controls_f, (x_ss, y_ss))


    q0 = hmodel.projection(m0, y0, p)

    q1 = hmodel.projection(m1, y1, p)

    exogenous = copy.deepcopy(hmodel.model.exogenous)
    _mc = exogenous.processes[1].discretize(to='mc', **hmodel.discretization_options)
    tmc = TrickyMarkovChain(q0, q1, _mc)


    dr1 = copy.deepcopy(eq.dr)
    dr1.set_values(x1)

    dr0 = time_iteration(hmodel.model, dr0=dr1, verbose=False, maxit=1, dprocess=tmc)
    s = dr0.endo_grid.nodes
    n_m = _mc.n_nodes
    xx0 = np.concatenate([e[None,:,:] for e in [dr0(i,s) for i in range(n_m)] ], axis=0)

    res_0 = xx0-x0

    grids = dr0.exo_grid, dr0.endo_grid

    aggres_0 = hmodel.𝒜(grids, m0, μ0, xx0, y0, p)

    return pack([res_0, aggres_0])


def get_derivatives(hmodel, eq):

    p = hmodel.calibration['parameters']
    states_ss = eq.states
    controls_ss = eq.controls


    g_s = jacobian(lambda u: G(hmodel, eq, u, controls_ss, p), states_ss)
    g_x = jacobian(lambda u: G(hmodel, eq, states_ss, u, p), controls_ss)
    g_e = np.zeros((g_s.shape[0], 1))
    f_s = jacobian(lambda u: F(hmodel, eq, u, controls_ss, states_ss, controls_ss, p), states_ss)
    f_x = jacobian(lambda u: F(hmodel, eq, states_ss, u, states_ss, controls_ss, p), controls_ss)
    f_S = jacobian(lambda u: F(hmodel, eq, states_ss, controls_ss, u, controls_ss, p), states_ss)
    f_X = jacobian(lambda u: F(hmodel, eq, states_ss, controls_ss, states_ss, u, p), controls_ss)

    return g_s, g_x, g_e, f_s, f_x, f_S, f_X

def perturb(hmodel, eq, verbose=True, return_system=False):

    from dolo.algos.perturbation import approximate_1st_order

    if verbose: print("Computing Jacobian...", end="")
    g_s, g_x, g_e, f_s, f_x, f_S, f_X = get_derivatives(hmodel, eq)
    if return_system:
        return g_s, g_x, g_e, f_s, f_x, f_S, f_X
    if verbose: print(colored("done", "green"))
    if verbose: print("Solving Perturbation...", end="")
    C0, evs = approximate_1st_order(g_s, g_x, g_e, f_s, f_x, f_S, f_X)
    if verbose: print(colored("done", "green"))
    C = C0
    P = g_s + g_x@C0

    return PerturbedEquilibrium(eq, C, P, evs)


class PerturbedEquilibrium:

    def __init__(hmodel, eq, C, P, evs):
        hmodel.eq = eq
        hmodel.C = C
        hmodel.P = P
        hmodel.evs = evs

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

    def simul(hmodel, dz):

        C = hmodel.C
        P = hmodel.P
        m_ss = hmodel.eq.m
        μ_ss = hmodel.eq.μ
        x_ss = hmodel.eq.x
        y_ss = hmodel.eq.y

        states_ss = hmodel.eq.states
        controls_ss = hmodel.eq.controls

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
