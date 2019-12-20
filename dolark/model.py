import dolo

from dolo.compiler.language import eval_data
from dolo.compiler.misc import CalibrationDict, calibration_to_vector
from dolo.numeric.processes import Conditional, ProductProcess, IIDProcess
from dolo.misc.display import read_file_or_url
import ruamel.yaml as ry
from dolo.compiler.model import Model
from dolo import time_iteration, improved_time_iteration

import dolang
from dolang.factory import FlatFunctionFactory

class AggregateException(Exception):
    pass


class HModel:

    def __init__(self, fname, i_options={}, dptype=None, debug=False):

        txt = read_file_or_url(fname)

        try:
            model_data, hmodel_data  = ry.load_all(txt, ry.RoundTripLoader)
        except Exception as ex:
            print ("Error while parsing YAML file. Probable YAML syntax error in file : ", fname )
            raise ex

        model_data, hmodel_data = ry.load_all(txt, Loader=ry.RoundTripLoader)

        self.__model__ = Model(model_data)
        self.data = hmodel_data

        self.discretization_options = i_options

        # cache for functions
        self.__equilibrium__ = None
        self.__projection__ = None
        self.__features__ = None

        self.debug = debug

        self.check()
        self.__set_changed__()

        from dolo.numeric.processes import IIDProcess, ProductProcess
        if dptype is None and isinstance(self.model.exogenous, ProductProcess) and (self.model.exogenous.processes[1], IIDProcess):
            dptype='iid'
        else:
            dptype='mc'
        self.dptype = dptype

    @property
    def agent(self):
        return self.__model__

    @property
    def model(self):
        return self.__model__

    def __set_changed__(self):
        # these depend on the calibration
        self.__exogenous__ = None
        self.__calibration__ = None

    def __get_calibration__(self):

        calibration = self.data.get("calibration", {})
        from dolo.compiler.triangular_solver import solve_triangular_system
        return solve_triangular_system(calibration)

    @property
    def features(self):
        if self.__features__ is None:
            __features__ = {}
            __features__['ex-ante-identical'] = not ('distribution' in self.data)
            __features__['conditional-processes'] = isinstance(self.model.exogenous, Conditional)
            __features__['iid-shocks'] = isinstance(self.model.exogenous, ProductProcess) and (False not in [isinstance(e, IIDProcess) for e in self.model.exogenous.processes[1:]])
            self.__features__ = __features__
        return self.__features__

    @property
    def name(self):
        return self.data.get("name", "unnamed")

    def check(self):

        kp = self.data.get('projection', None)
        if kp is None:
            raise AggregateException("Missing 'projection section'.")

        idiosyms = self.model.symbols['exogenous']
        vals = [*kp.keys()]

        targets = [e for e in idiosyms if e in vals]
        if len(targets)<len(vals):
            diff = set(vals).difference(set(targets))
            raise AggregateException(f"Some projected values were not defined as exogenous in the agent's program: {', '.join(diff)}")
        expected = idiosyms[:len(targets)]
        if tuple(targets) != tuple(expected):
            raise AggregateException(f"Projected values must match first exogenous variables of model. Found {', '.join(targets)}. Expected {', '.join(expected)}")

    def set_calibration(self, *pargs, **kwargs):
        if len(pargs)==1:
            self.set_calibration(**pargs[0])
        self.__set_changed__()
        self.data['calibration'].update(kwargs)

    @property
    def symbols(self):
        symbols = {sg: [*self.data['symbols'][sg]] for sg in self.data['symbols'].keys()}
        return symbols

    @property
    def variables(self):
        return sum([self.symbols[v] for v in ['exogenous','aggregate']], [])

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = self.__get_calibration__()
            calib = calibration_to_vector(self.symbols, calibration_dict)
            self.__calibration__ = CalibrationDict(self.symbols, calib)  #
        return self.__calibration__

    @property
    def exogenous(self):

        # old style only
        calib = self.calibration.flat
        return eval_data(self.data['exogenous'], calibration=calib)


    @property
    def distribution(self):

        # old style only
        calib = self.calibration.flat
        if 'distribution' in self.data:
            return eval_data(self.data['distribution'], calibration=calib)
        else:
            return None

    @property
    def projection(self): #, m: 'n_e', y: "n_y", p: "n_p"):

        if self.__projection__ is None:

            arguments_ = {
                # 'e': [(e,0) for e in self.model.symbols['exogenous']],
                # 's': [(e,0) for e in self.model.symbols['states']],
                # 'x': [(e,0) for e in self.model.symbols['controls']],
                'm': [(e,0) for e in self.symbols['exogenous']],
                'y': [(e,0) for e in self.symbols['aggregate']],
                'p': self.symbols['parameters']
            }

            vars = sum( [[e[0] for e in h] for h in [*arguments_.values()][:-1]], [])

            arguments = {k: [dolang.symbolic.stringify_symbol(e) for e in v] for k,v in arguments_.items()}

            preamble = {} # for now

            projdefs = self.data.get('projection', {})
            pkeys = [*projdefs.keys()]
            n_p = len(pkeys)
            equations = [projdefs[v] for v in self.model.symbols['exogenous'][:n_p]]
            equations = [dolang.stringify(eq, variables=vars) for eq in equations]
            content = {f'{pkeys[i]}_0': eq for i, eq in enumerate(equations)}
            fff = FlatFunctionFactory(preamble, content, arguments, 'equilibrium')
            fun = dolang.function_compiler.make_method_from_factory(fff, debug=self.debug)
            from dolang.vectorize import standard_function
            self.__projection__ = standard_function(fun[1], len(equations))

        return self.__projection__


    @property
    def ℰ(self):

        if self.__equilibrium__ is None:

            arguments_ = {
                'e': [(e,0) for e in self.model.symbols['exogenous']],
                's': [(e,0) for e in self.model.symbols['states']],
                'x': [(e,0) for e in self.model.symbols['controls']],
                'm': [(e,0) for e in self.symbols['exogenous']],
                'y': [(e,0) for e in self.symbols['aggregate']],
                'p': self.symbols['parameters']
            }

            vars = sum( [[e[0] for e in h] for h in [*arguments_.values()][:-1]], [])

            arguments = {k: [dolang.symbolic.stringify_symbol(e) for e in v] for k,v in arguments_.items()}

            preamble = {} # for now

            equations = [("{}-({})".format(*(str(eq).split('='))) if '=' in eq else eq) for eq in self.data['equilibrium']]
            equations = [dolang.stringify(eq, variables=vars) for eq in equations]
            content = {f'eq_{i}': eq for i, eq in enumerate(equations)}
            fff = FlatFunctionFactory(preamble, content, arguments, 'equilibrium')
            fun = dolang.function_compiler.make_method_from_factory(fff, debug=self.debug)
            from dolang.vectorize import standard_function
            self.__equilibrium__ = standard_function(fun[1], len(equations))

        return self.__equilibrium__

    def τ(self, m, p):
        # exogenous process is assumed to be deterministic
        # TEMP:  works only if exogenous is an AR1
        ρ = self.exogenous.rho
        return ρ*m

    def 𝒜(self, grids, m0: 'n_e', μ0: "n_m.N" , xx0: "n_m.N.n_x", y0: "n_y", p: "n_p"):

        from dolo.numeric.processes import EmptyGrid
        import numpy as np
        μ0 = np.array(μ0)
        ℰ = self.ℰ
        exg, eng = grids
        if isinstance(exg, EmptyGrid):
            # this is so sad
            mi = self.model.calibration['exogenous'][None,:] # not used anyway...
        else:
            mi = exg.nodes()
        s = eng.nodes()
        res = sum( [μ0[i,:] @ ℰ(mi[i,:],s,xx0[i,:,:],m0,y0,p) for i in range(xx0.shape[0]) ])
        return res


    def get_starting_rule(self, method='improved_time_iteration', **kwargs):
        # provides initial guess for d.r. by solving agent's problem

        mc = self.model.exogenous.discretize(to='mc', options=[{},self.discretization_options])
        # dr0 = time_iteration(self.model, dprocess=mc)
        if method=='improved_time_iteration':
            sol = improved_time_iteration(self.model, dprocess=mc, **kwargs)
        elif method=='time_iteration':
            sol = time_iteration(self.model, dprocess=mc, **kwargs)
        dr0 = sol.dr

        return dr0
