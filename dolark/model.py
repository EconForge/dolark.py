import dolo

from dolo.compiler.language import eval_data
from dolo.compiler.misc import CalibrationDict, calibration_to_vector

from dolo.misc.display import read_file_or_url
import ruamel.yaml as ry
from dolo.compiler.model import Model

from dolark.perturbation import AggregateModel

import dolang
from dolang.factory import FlatFunctionFactory
from dolang.symbolic import sanitize

class AggregateException(Exception):
    pass

# deriving from AggregateModel is a very bad idea
class HModel(AggregateModel):

    def __init__(self, fname, i_options={}):

        txt = read_file_or_url(fname)

        try:
            model_data, hmodel_data  = ry.load_all(txt, ry.RoundTripLoader)
        except Exception as ex:
            print ("Error while parsing YAML file. Probable YAML syntax error in file : ", fname )
            raise ex

        model_data, hmodel_data = ry.load_all(txt, Loader=ry.RoundTripLoader)

        self.model = Model(model_data)
        self.data = hmodel_data

        self.discretization_options = i_options

        self.__equilibrium__ = None
        self.check()
        self.__set_changed__()

        # probably not optimal
        super().__init__(self.model)

    def __set_changed__(self):
        # these depend on the calibration
        self.__exogenous__ = None
        self.__calibration__ = None

    def __get_calibration__(self):

        symbols = self.symbols
        calibration = self.data.get("calibration", {})
        from dolo.compiler.triangular_solver import solve_triangular_system
        return solve_triangular_system(calibration)


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

            vars = sum( [[e[0] for e in h] for h in arguments_.values()], [])

            arguments = {k: [dolang.symbolic.stringify_symbol(e) for e in v] for k,v in arguments_.items()}

            preamble = {} # for now

            equations = [("{}-({})".format(*(str(eq).split('='))) if '=' in eq else eq) for eq in self.data['equilibrium']]
            equations = [dolang.stringify(eq, variables=vars) for eq in equations]
            content = {f'eq_{i}': eq for i, eq in enumerate(equations)}
            fff = FlatFunctionFactory(preamble, content, arguments, 'equilibrium')
            fun = dolang.function_compiler.make_method_from_factory(fff)
            self.__equilibrium__ = fun[1]

        return self.__equilibrium__

    def τ(self, m, p):
        # exogenous process is assumed to be deterministic
        # TEMP:  works only if exogenous is an AR1
        ρ = self.exogenous.rho
        return ρ*m

    def 𝒜(self, m0: 'n_e', μ0: "n_m.N" , xx0: "n_m.N.n_x", y0: "n_y", p: "n_p"):

        import numpy as np
        μ0 = np.array(μ0)
        ℰ = self.ℰ
        exg, eng = self.grids
        mi = exg.nodes()
        s = eng.nodes()
        res = sum( [μ0[i,:] @ ℰ(mi[i,:],s,xx0[i,:,:],m0,y0,p) for i in range(xx0.shape[0]) ])
        return res
