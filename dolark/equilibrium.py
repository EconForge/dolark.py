import scipy
from dolo import colored
import numpy as np
import pandas as pd
from .shocks import inject_process
from dolo import improved_time_iteration, time_iteration, ergodic_distribution

from .shocks import discretize_idiosyncratic_shocks


class Equilibrium:

    def __init__(self, aggmodel, m, μ, dr, X, S=None):
        self.m = m
        self.μ = μ
        self.dr = dr
        self.x = np.concatenate(
            [
                e[None, :, :]
                for e in [
                    dr(i, dr.endo_grid.nodes)
                    for i in range(max(dr.exo_grid.n_nodes, 1))
                ]
            ],
            axis=0,
        )
        self.X = X
        self.S = S
        self.c = dr.coefficients

        self.controls = np.concatenate([e.ravel() for e in (self.x, X)])
        if aggmodel.features['with-aggregate-states']:
            self.states = np.concatenate([e.ravel() for e in (m, μ, S)])
        else:
            self.states = np.concatenate([e.ravel() for e in (m, μ)])
        self.aggmodel = aggmodel

    # backward compatibility
    @property
    def y(self):
        return self.X

    def as_df(self):
        model = self.aggmodel.model
        eq = self
        exg = np.column_stack([range(eq.dr.exo_grid.n_nodes), eq.dr.exo_grid.nodes])
        edg = np.column_stack([eq.dr.endo_grid.nodes])
        N_m = exg.shape[0]
        N_s = edg.shape[0]
        ssg = np.concatenate(
            [exg[:, None, :].repeat(N_s, axis=1), edg[None, :, :].repeat(N_m, axis=0)],
            axis=2,
        ).reshape((N_m * N_s, -1))
        x = np.concatenate(
            [eq.dr(i, edg) for i in range(max(eq.dr.exo_grid.n_nodes, 1))], axis=0
        )
        import pandas as pd

        cols = (
            ["i_m"]
            + model.symbols["exogenous"]
            + model.symbols["states"]
            + ["μ"]
            + model.symbols["controls"]
        )
        df = pd.DataFrame(np.column_stack([ssg, eq.μ.ravel(), x]), columns=cols)
        return df


def transition_residual(
    hmodel,
    m0: "vector",
    S0: "vector",
    X0: "vector",
    p=None,
):

    if hmodel.features["with-aggregate-states"]:
        if p is None:
            p = hmodel.calibration["parameters"]
        return S0 - hmodel.𝒢(m0, S0, X0, m0, p)
    else:
        raise Exception(
            "The considered model does not include any valid aggregate transition equation."
        )


def equilibrium(
    hmodel,
    m0: "vector",
    S0=None,
    X0: "vector"=None,
    p=None,
    dr0=None,
    grids=None,
    verbose=False,
    return_equilibrium=True,
):
    if p is None:
        p = hmodel.calibration["parameters"]

    if S0 is None:
        q0 = hmodel.projection(m0, X0, p)
    else:
        q0 = hmodel.projection(m0, S0, X0, p)


    dp = inject_process(q0, hmodel.model.exogenous)

    sol = time_iteration(hmodel.model, dr0=dr0, dprocess=dp, maxit=10, verbose=verbose)
    sol = improved_time_iteration(hmodel.model, dr0=sol, dprocess=dp, verbose=verbose)
    dr = sol.dr

    if grids is None:
        exg, edg = grids = dr.exo_grid, dr.endo_grid
    else:
        exg, edg = grids

    Π0, μ0 = ergodic_distribution(hmodel.model, dr, exg, edg, dp)

    s = edg.nodes
    if exg.n_nodes == 0:
        nn = 1
        μμ0 = μ0.data[None, :]
    else:
        nn = exg.n_nodes
        μμ0 = μ0.data

    xx0 = np.concatenate([e[None, :, :] for e in [dr(i, s) for i in range(nn)]], axis=0)
    res = hmodel.𝒜(grids, m0, μμ0, xx0, X0, m0, X0, p, S0=S0, S1=S0)
    if return_equilibrium:
        return (res, sol, μ0, Π0)
    else:
        return res


def find_steady_state(hmodel, dr0=None, verbose=True, distribs=None, return_fun=False):

    m0 = hmodel.calibration["exogenous"]
    X0 = hmodel.calibration["aggregate"]
    p = hmodel.calibration["parameters"]

    if dr0 is None:
        if verbose:
            print("Computing Initial Rule... ", end="")
        dr0 = hmodel.get_starting_rule()
        if verbose:
            print(colored("done", "green"))

    if verbose:
        print("Computing Steady State...", end="")

    if distribs is None:
        dist = [(1.0, {})]
        if not hmodel.features["ex-ante-identical"]:
            dist = distribs = discretize_idiosyncratic_shocks(hmodel.distribution)
    else:
        dist = distribs

    if hmodel.features["with-aggregate-states"]:
        S0 = hmodel.calibration["states"]
        n_S = len(S0)
        n_X = len(X0)

        def fun(u):
            res_X = X0 * 0
            for w, kwargs in dist:
                hmodel.model.set_calibration(**kwargs)
                res_X += w * equilibrium(
                    hmodel, m0, S0=u[:n_S], X0=u[n_S:], dr0=dr0, return_equilibrium=False
                )
            res_S = transition_residual(hmodel, m0, u[:n_S], u[n_S:])
            res = np.concatenate((res_S, res_X))
            if verbose=='full':
                print(f"Value at {u} | {res}")
            return res

        Y0 = np.concatenate((S0, X0))
        if return_fun:
            return (fun, Y0)

        solution = scipy.optimize.root(fun, x0=Y0)
    else:

        def fun(u):
            res = X0 * 0
            for w, kwargs in dist:
                hmodel.model.set_calibration(**kwargs)
                res += w * equilibrium(hmodel, m0, X0=u, dr0=dr0, return_equilibrium=False)
            if verbose=='full':
                print(f"Value at {u} | {res}")
            return res

        if return_fun:
            return (fun, X0)

        solution = scipy.optimize.root(fun, x0=X0)

    if not solution.success:
        if verbose:
            print(colored("failed", "red"))
    else:
        if verbose:
            print(colored("done", "green"))

    # grid_m = model.exogenous.discretize(to='mc', options=[{},{'N':N_mc}]).nodes
    # grid_s = model.get_grid().nodes
    #
    Y_ss = solution.x  # vector of aggregate endogenous variables
    m_ss = m0  # vector fo aggregate exogenous
    eqs = []
    if hmodel.features["with-aggregate-states"]:
        for w, kwargs in dist:
            hmodel.model.set_calibration(**kwargs)
            (res_ss, sol_ss, μ_ss, Π_ss) = equilibrium(
                hmodel,
                m_ss,
                X0=Y_ss[n_S:],
                p=p,
                dr0=dr0,
                S0=Y_ss[:n_S],
                return_equilibrium=True,
            )
            μ_ss = μ_ss.data
            dr_ss = sol_ss.dr
            eqs.append(
                [
                    w,
                    Equilibrium(
                        hmodel, m_ss, μ_ss, sol_ss.dr, Y_ss[:n_S], S=Y_ss[:n_S]
                    ),
                ]
            )
    else:
        for w, kwargs in dist:
            hmodel.model.set_calibration(**kwargs)
            (res_ss, sol_ss, μ_ss, Π_ss) = equilibrium(
                hmodel, m_ss, X0=Y_ss, p=p, dr0=dr0, return_equilibrium=True
            )
            μ_ss = μ_ss.data
            dr_ss = sol_ss.dr
            eqs.append([w, Equilibrium(hmodel, m_ss, μ_ss, sol_ss.dr, Y_ss)])

    if distribs is None:
        return eqs[0][1]
    else:
        return eqs
