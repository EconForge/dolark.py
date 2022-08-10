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


def G(hmodel, equilibrium, exo_p, states_p, controls_p, exo, p):

    eq = equilibrium
    m_ss = eq.m
    μ_ss = eq.μ
    x_ss = eq.x
    y_ss = eq.y
    S_ss = eq.S

    m0 = exo_p
    m1 = exo

    if hmodel.features["with-aggregate-states"]:
        μ0, S0 = unpack(states_p, (μ_ss, S_ss))
    else:
        (μ0,) = unpack(states_p, (μ_ss,))

    x0, y0 = unpack(controls_p, (x_ss, y_ss))

    X0 = y0  # TODO: unmessify

    if hmodel.features["with-aggregate-states"]:
        q0 = hmodel.projection(m0, S0, y0, p)
        q1 = hmodel.projection(m1, S0, y0, p)  # does it make sense?
    else:
        q0 = hmodel.projection(m0, y0, p)
        q1 = hmodel.projection(m1, y0, p)

    exogenous = copy.deepcopy(hmodel.model.exogenous)
    _mc = exogenous.processes[1].discretize(to="mc", **hmodel.discretization_options)

    mc = TrickyMarkovChain(q0, q1, _mc)

    dr0 = copy.deepcopy(eq.dr)
    dr0.set_values(x0)

    exg, edg = dr0.exo_grid, dr0.endo_grid
    # we should not compute μ here...
    μ0 = μ0.ravel()
    Π0, _ = ergodic_distribution(hmodel.model, dr0, exg, edg, mc)

    k = len(μ0)
    Π0 = Π0.reshape((k, k))

    # m1 = hmodel.τ(m0, p)
    μ1 = μ0 @ Π0

    if hmodel.features["with-aggregate-states"]:
        S1 = hmodel.𝒢(m0, S0, X0, m1, p)
        return pack([μ1, S1])
    else:
        return pack([μ1])


def F(hmodel, equilibrium, m, states, controls, m_f, states_f, controls_f, p):

    eq = equilibrium
    m_ss = eq.m
    μ_ss = eq.μ
    x_ss = eq.x
    y_ss = eq.y

    m0 = m
    m1 = m_f

    if hmodel.features["with-aggregate-states"]:
        μ0, S0 = unpack(states, (μ_ss, eq.S))
        μ1, S1 = unpack(states_f, (μ_ss, eq.S))
    else:
        (μ0,) = unpack(states, (μ_ss,))
        (μ1,) = unpack(states_f, (μ_ss,))

    x0, y0 = unpack(controls, (x_ss, y_ss))
    x1, y1 = unpack(controls_f, (x_ss, y_ss))

    if hmodel.features["with-aggregate-states"]:
        q0 = hmodel.projection(m0, S0, y0, p)
        q1 = hmodel.projection(m1, S1, y1, p)
    else:
        q0 = hmodel.projection(m0, y0, p)
        q1 = hmodel.projection(m1, y1, p)

    exogenous = copy.deepcopy(hmodel.model.exogenous)
    _mc = exogenous.processes[1].discretize(to="mc", **hmodel.discretization_options)
    tmc = TrickyMarkovChain(q0, q1, _mc)

    dr1 = copy.deepcopy(eq.dr)
    dr1.set_values(x1)

    sol0 = time_iteration(hmodel.model, dr0=dr1, verbose=False, maxit=1, dprocess=tmc)
    dr0 = sol0.dr
    s = dr0.endo_grid.nodes
    n_m = _mc.n_nodes
    xx0 = np.concatenate(
        [e[None, :, :] for e in [dr0(i, s) for i in range(n_m)]], axis=0
    )

    res_0 = xx0 - x0

    grids = dr0.exo_grid, dr0.endo_grid

    # m0: "n_e",
    # μ0: "n_m.N",
    # xx0: "n_m.N.n_x",
    # X0: "n_X",
    # m1: "n_e",
    # X1: "n_X",
    # p: "n_p",
    # S0=None,
    # S1=None

    if hmodel.features["with-aggregate-states"]:
        aggres_0 = hmodel.𝒜(grids, m0, μ0, xx0, y0, m1, y1, p, S0=S0, S1=S1)
    else:
        aggres_0 = hmodel.𝒜(grids, m0, μ0, xx0, y0, m1, y1, p)

    return pack([res_0, aggres_0])


def get_derivatives(hmodel, eq):

    s = eq.states
    x = eq.controls
    m = eq.m
    p = hmodel.calibration["parameters"]

    test1 = G(hmodel, eq, m, s, x, m, p)
    test2 = F(hmodel, eq, m, s, x, m, s, x, p)

    # TODO: generalize this part
    h_m = hmodel.exogenous.ρ

    g_m = jacobian(lambda u: G(hmodel, eq, u, s, x, m, p), m)
    g_s = jacobian(lambda u: G(hmodel, eq, m, u, x, m, p), s)
    g_x = jacobian(lambda u: G(hmodel, eq, m, s, u, m, p), x)
    g_M = jacobian(lambda u: G(hmodel, eq, m, s, x, u, p), m)

    f_m = jacobian(lambda u: F(hmodel, eq, u, s, x, m, s, x, p), m)
    f_s = jacobian(lambda u: F(hmodel, eq, m, u, x, m, s, x, p), s)
    f_x = jacobian(lambda u: F(hmodel, eq, m, s, u, m, s, x, p), x)
    f_M = jacobian(lambda u: F(hmodel, eq, m, s, x, u, s, x, p), m)
    f_S = jacobian(lambda u: F(hmodel, eq, m, s, x, m, u, x, p), s)
    f_X = jacobian(lambda u: F(hmodel, eq, m, s, x, m, s, u, p), x)

    return FirstOrderModel(h_m, g_m, g_s, g_x, g_M, f_m, f_s, f_x, f_M, f_S, f_X)


from dataclasses import dataclass


class Matrix:
    pass


@dataclass
class FirstOrderModel:

    # this represents a model:
    #
    # exogenous:  m_t = h(m_{t-1}) + \epsilon_t
    # endogenous: s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t, s_t, x_t)
    # arbitrage:  f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1})

    h_m: Matrix
    g_m: Matrix
    g_s: Matrix
    g_x: Matrix
    g_M: Matrix
    f_m: Matrix
    f_s: Matrix
    f_x: Matrix
    f_M: Matrix
    f_S: Matrix
    f_X: Matrix


def solve_fom(fom: FirstOrderModel):

    from dolo.algos.perturbation import approximate_1st_order

    g_m = fom.g_m + fom.g_M * fom.h_m
    g_s = fom.g_s
    g_x = fom.g_x
    g_e = fom.g_M
    f_m = fom.f_m
    f_s = fom.f_s
    f_x = fom.f_x
    f_M = fom.f_M
    f_S = fom.f_S
    f_X = fom.f_X

    n_s = g_s.shape[1]
    n_x = g_x.shape[1]
    n_m = g_m.shape[1]

    I = lambda p: np.eye(p)
    Z = lambda p, q: np.zeros((p, q))

    # now we express the system with (m,s) as endogenous states
    G_s = np.row_stack(
        [np.column_stack([fom.h_m, Z(n_m, n_s)]), np.column_stack([g_m, g_s])]
    )

    G_x = np.row_stack([Z(n_m, n_x), g_x])

    G_e = np.row_stack([I(n_m), Z(n_s, n_m)])

    F_s = np.column_stack([f_m, f_s])
    F_x = f_x
    F_S = np.column_stack([f_M, f_S])
    F_X = f_X

    C, evs = approximate_1st_order(G_s, G_x, G_e, F_s, F_x, F_S, F_X)

    return C[:, :n_m], C[:, n_m:], evs


def perturb(hmodel, eq, verbose=True, return_system=False):

    # get first order model
    if verbose:
        print("Computing Jacobian...", end="")
    fom = get_derivatives(hmodel, eq)
    if verbose:
        print(colored("done", "green"))
    if verbose:
        print("Solving Perturbation...", end="")

    C_m, C_s, evs = solve_fom(fom)

    if verbose:
        print(colored("done", "green"))

    return PerturbedEquilibrium(eq, fom, C_m, C_s, evs)


class PerturbedEquilibrium:
    def __init__(self, eq, fom, C_m, C_s, evs):

        self.fom = fom
        self.eq = eq
        self.C_m = C_m
        self.C_s = C_s
        self.evs = evs

    def response(self, T, m0=None):
        return self.simulate(T, m0=m0, stochastic=True)

    def simulate(self, T, m0=None, s0=None, stochastic=True):

        import xarray

        hmodel = self.eq.aggmodel

        C_m = self.C_m
        C_s = self.C_s

        n_m = self.fom.g_m.shape[1]  # exogenous states
        n_s = self.fom.g_s.shape[1]  # endogenous states
        n_x = self.fom.g_x.shape[1]

        g_m = self.fom.g_m
        g_s = self.fom.g_s
        g_x = self.fom.g_x
        g_M = self.fom.g_M

        m_sim = hmodel.exogenous.simulate(1, m0=m0, T=T + 1, stochastic=stochastic)
        m_sim = m_sim[:, 0, :]

        s_sim = np.zeros((T + 1, n_s))
        x_sim = np.zeros((T + 1, n_x))

        if s0 is not None:
            raise Exception("Not implemented yet.")
        #         s_sim[0,:] =

        x_sim[0, :] = C_m @ m_sim[0, :] + C_s @ s_sim[0, :]

        for t in range(1, T):
            s_sim[t, :] = (
                g_m @ m_sim[t - 1, :]
                + g_s @ s_sim[t - 1, :]
                + g_x @ x_sim[t - 1, :]
                + g_M @ m_sim[t, :]
            )
            x_sim[t, :] = C_m @ m_sim[t, :] + C_s @ s_sim[t, :]

        # add steady-state

        m_sim = m_sim[:, :] + self.eq.m[None, :]
        s_sim = s_sim[:, :] + self.eq.states[None, :]
        x_sim = x_sim[:, :] + self.eq.controls[None, :]

        m_sim = xarray.DataArray(
            m_sim,
            coords=[("T", [*range(0, T + 1)]), ("V", hmodel.symbols["exogenous"])],
        )
        aggvars = hmodel.symbols["aggregate"]
        sn = [f"_x_{i}" for i in range(n_x - len(aggvars))] + aggvars
        x_sim = xarray.DataArray(x_sim, coords=[("T", [*range(0, T + 1)]), ("V", sn)])

        if hmodel.features["with-aggregate-states"]:
            aggstates = hmodel.symbols["states"]
            sn = [f"_μ_{i}" for i in range(n_s - len(aggstates))] + aggstates
        else:
            sn = [f"_μ_{i}" for i in range(n_s)]

        s_sim = xarray.DataArray(s_sim, coords=[("T", [*range(0, T + 1)]), ("V", sn)])

        return xarray.concat([m_sim, s_sim, x_sim], dim="V")
