from dolo.numeric.grids import *
from dolo.numeric.decision_rule import CallableDecisionRule, cat_grids
import numpy as np
import tqdm

class Linear:
    pass

class Cubic:
    pass

class Chebychev:
    pass

interp_methods = {
    'cubic': Cubic(),
    'linear': Linear(),
    'multilinear': Linear(),
    'chebychev': Chebychev()
}

# we keep the same user-facing-API

class CallableDecisionRule:

    def __call__(self, *args):
        args = [np.array(e) for e in args]
        if len(args)==1:
            return self.eval_s(args[0])
        elif len(args)==2:
            if args[0].dtype in ('int64','int32'):
                (i,s) = args
                if s.ndim == 1:
                    return self.eval_is(i, s[None,:])[0,:]
                else:
                    return self.eval_is(i, s)
                return self.eval_is()
            else:
                (m,s) = args[0],args[1]
                if s.ndim == 1 and m.ndim == 1:
                    return self.eval_ms(m[None,:], s[None,:])[0,:]
                elif m.ndim == 1:
                    m = m[None,:]
                elif s.ndim == 1:
                    s = s[None,:]
                return self.eval_ms(m,s)


class DecisionRule(CallableDecisionRule):

    exo_grid: Grid
    endo_grid: Grid

    def __init__(self, exo_grid: Grid, endo_grid: Grid, interp_method='cubic', dprocess=None, values=None):

        if interp_method not in interp_methods.keys():
            raise Exception(f"Unknown interpolation type: {interp_method}. Try one of: {tuple(interp_methods.keys())}")

        self.exo_grid = exo_grid
        self.endo_grid = endo_grid
        self.interp_method = interp_method

        self.__interp_method__ = interp_methods[interp_method]

        args = (self, exo_grid, endo_grid, interp_methods[interp_method])

        try:
            aa = args + (None, None)
            fun = eval_ms[tuple(map(type, aa))]
            self.__eval_ms__ = fun
        except Exception as exc:
            pass

        try:
            aa = args + (None, None)
            fun = eval_is[tuple(map(type, aa))]
            self.__eval_is__ = fun
        except Exception as exc:
            pass

        try:
            aa = args + (None, None)
            fun = eval_s[tuple(map(type, aa))]
            self.__eval_s__ = fun
        except Exception as exc:
            pass

        fun = get_coefficients[tuple(map(type, args))]
        self.__get_coefficients__ = fun

        if values is not None:
            self.set_values(values)

    def set_values(self, x):
        self.coefficients = self.__get_coefficients__(self, self.exo_grid, self.endo_grid, self.__interp_method__, x)

    def eval_ms(self, m, s):
        return self.__eval_ms__(self, self.exo_grid, self.endo_grid, self.__interp_method__, m, s)

    def eval_is(self, i, s):
        return self.__eval_is__(self, self.exo_grid, self.endo_grid, self.__interp_method__, i, s)

    def eval_s(self, s):
        return self.__eval_s__(self, self.exo_grid, self.endo_grid, self.__interp_method__, s)


# this is *not* meant to be used by users

from multimethod import multimethod

# Cartesian x Cartesian x Linear

@multimethod
def get_coefficients(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, x):
    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    xx = x.reshape(tuple(grid.n)+(-1,))
    return xx


@multimethod
def eval_ms(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, m, s):

    assert(m.ndim==s.ndim==2)

    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    coeffs = itp.coefficients
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    from interpolation.splines import eval_linear

    x = np.concatenate([m, s], axis=1)

    return eval_linear(gg, coeffs, x)


@multimethod
def eval_is(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, i, s):
    m = exo_grid.node(i)[None,:]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)


# Cartesian x Cartesian x Cubic

@multimethod
def get_coefficients(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, x):

    from interpolation.splines.prefilter_cubic import prefilter_cubic
    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    x = x.reshape(tuple(grid.n)+(-1,))
    d = len(grid.n)
    # this gg could be stored as a member of itp
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    return prefilter_cubic(gg, x)

@multimethod
def eval_ms(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, m, s):

    from interpolation.splines import eval_cubic

    assert(m.ndim==s.ndim==2)

    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    coeffs = itp.coefficients
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    x = np.concatenate([m, s], axis=1)

    return eval_cubic(gg, coeffs, x)


@multimethod
def eval_is(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, i, s):
    m = exo_grid.node(i)[None,:]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)



# UnstructuredGrid x Cartesian x Linear

@multimethod
def get_coefficients(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, x):
    return [x[i].copy() for i in range(x.shape[0])]

@multimethod
def eval_is(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, i, s):

    from interpolation.splines import eval_linear
    assert(s.ndim==2)

    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients[i]
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_linear(gg, coeffs, s)


# UnstructuredGrid x Cartesian x Cubic

@multimethod
def get_coefficients(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Cubic, x):
    from interpolation.splines.prefilter_cubic import prefilter_cubic
    grid = endo_grid # one single CartesianGrid
    d = len(grid.n)
    # this gg could be stored as a member of itp
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    return [prefilter_cubic(gg, x[i]) for i in range(x.shape[0])]


@multimethod
def eval_is(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Cubic, i, s):

    from interpolation.splines import eval_cubic
    assert(s.ndim==2)

    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients[i]
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_cubic(gg, coeffs, s)

###### Test

if __name__ == "__main__":

    exo_grid = CartesianGrid([0.0], [1.0], [10])
    endo_grid = CartesianGrid([0.0], [1.0], [10])
    values = np.random.random((10,10,2))

    m = np.linspace(0,1, 100)[:,None]
    s = np.linspace(0,1, 100)[:,None]


    dr_lin = DecisionRule(exo_grid, endo_grid, values=values, interp_method='linear')

    # evaluation stricte: m et s doivent etre deux matrices
    dr_lin.eval_ms(m,s)

    # evaluation souple: m et s peuvent etre des vecteurs, dispatch sur le type de m
    dr_lin(m[0,:],s[0,:])

    dr_lin(0,s[0,:])



    dr_spline = DecisionRule(exo_grid, endo_grid, values=values, interp_method='cubic')

    # evaluation stricte: m et s doivent etre deux matrices
    dr_spline.eval_ms(m,s)

    # evaluation souple: m et s peuvent etre des vecteurs, dispatch sur le type de m
    dr_spline(m[0,:],s[0,:])
    dr_spline(0,s[0,:])


    # Unstructured exogenous grid

    exo_grid = UnstructuredGrid(np.array([[0.2, 0.5, 0.7]]))
    endo_grid = CartesianGrid([0.0], [1.0], [10])
    values = np.random.random((3,10,2))

    s = np.linspace(-0.1,1, 100)[:,None]

    dr_mc_lin = DecisionRule(exo_grid, endo_grid, values=values, interp_method='linear')

    dr_mc_lin.eval_is(0,s)
    dr_mc_lin(0,s[0,:])
    dr_mc_lin(1,s[0,:])



    dr_mc_cub = DecisionRule(exo_grid, endo_grid, values=values, interp_method='cubic')

    dr_mc_cub.eval_is(0,s)
    dr_mc_cub(0,s[0,:])
    dr_mc_cub(1,s[0,:])



from dataclasses import dataclass

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
        B = -(f_s + f_S@g_s + f_X@C@g_s)
        A = (f_x + f_S@g_x + f_X@C@g_x)
        C1 = numpy.linalg.solve(A,B)
        η = abs(C1-C).max()
        C = C1
        # errors.append(ϵ)

    ϵ = abs(f_s + f_x@C + f_S@(g_s+g_x@C) + f_X@C@(g_s+g_x@C)).max()

    P = g_s + g_x@C
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
        l.append(v[t:t+aa].reshape(a.shape))
        t+=aa
    return l

def fun(v):
    arg = unpack(v, args)
    return np.concatenate( [e.ravel() for e in system(*arg) ])

import numpy as np


def jacobian(f, x, dx=10e-6):
    f0 = f(x)
    n = len(f0)
    p = len(x)
    jac = np.zeros((n, p))
    for j in range(p): #through columns to allow for vector addition
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = np.row_stack([(xi if k != j else xi+Dxj) for k, xi in enumerate(x)])
        jac[:, j] = (f(x_plus)-f0)/Dxj
    return jac

from dolo.numeric.processes import MarkovChain

class TrickyMarkovChain(MarkovChain):

    def __init__(self, μ1, μ2, mc):
        self.μ1 = np.array(μ1)
        self.μ2 = np.array(μ2)
        self.mc = mc
        self.transitions = mc.transitions
        self.values = np.column_stack([self.μ1[None,:].repeat(mc.values.shape[0], axis=0), mc.values])

    def inode(self, i:int, j:int): # vector
        val = self.values[j,:].copy()
        val[:len(self.μ2)] = self.μ2
        return val
