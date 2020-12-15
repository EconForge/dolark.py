from dolo.numeric.grids import *
import numpy as np
import tqdm

from dataclasses import dataclass

######


@dataclass
class PerturbationResult:
    C: np.array
    P: np.array
    tol_η: float
    tol_ϵ: float


def newtonator(g_s, g_x, f_s, f_x, f_S, f_X, tol_η=1e-6, tol_ϵ=1e-6, maxit=10000):

    import numpy.linalg

    import scipy.linalg

    # errors = []
    n_s = g_s.shape[1]
    n_x = g_x.shape[1]
    C = np.random.random((n_x, n_s))

    for i in range(maxit):
        B = -(f_s + f_S @ g_s + f_X @ C @ g_s)
        A = f_x + f_S @ g_x + f_X @ C @ g_x
        C1 = numpy.linalg.solve(A, B)
        η = abs(C1 - C).max()
        C = C1
        # errors.append(ϵ)

    ϵ = abs(f_s + f_x @ C + f_S @ (g_s + g_x @ C) + f_X @ C @ (g_s + g_x @ C)).max()

    P = g_s + g_x @ C
    evs = scipy.linalg.eig(P)[0]

    # assert( abs(evs).max()<1)

    return PerturbationResult(C, P, η, ϵ)


def pack(v):
    return np.concatenate([e.ravel() for e in v])


def unpack(v, args):
    t = 0
    l = []
    for a in args:
        aa = np.prod(np.array(a).shape)
        l.append(v[t : t + aa].reshape(a.shape))
        t += aa
    return l


def fun(v):
    arg = unpack(v, args)
    return np.concatenate([e.ravel() for e in system(*arg)])


import numpy as np


def jacobian(f, x, dx=10e-6):
    f0 = f(x)
    n = len(f0)
    p = len(x)
    jac = np.zeros((n, p))
    for j in range(p):  # through columns to allow for vector addition
        Dxj = abs(x[j]) * dx if x[j] != 0 else dx
        x_plus = np.row_stack([(xi if k != j else xi + Dxj) for k, xi in enumerate(x)])
        jac[:, j] = (f(x_plus) - f0) / Dxj
    return jac


from dolo.numeric.processes import MarkovChain


class TrickyMarkovChain(MarkovChain):
    def __init__(self, μ1, μ2, mc):
        self.μ1 = np.array(μ1)
        self.μ2 = np.array(μ2)
        self.mc = mc
        self.transitions = mc.transitions
        self.values = np.column_stack(
            [self.μ1[None, :].repeat(mc.values.shape[0], axis=0), mc.values]
        )

    def inode(self, i: int, j: int):  # vector
        val = self.values[j, :].copy()
        val[: len(self.μ2)] = self.μ2
        return val
