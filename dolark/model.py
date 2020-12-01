import dolo

from dolo.compiler.misc import CalibrationDict, calibration_to_vector
from dolo.numeric.processes import Conditional, ProductProcess, IIDProcess
from dolo.misc.display import read_file_or_url
import yaml
from dolo.compiler.model import Model
from dolo import time_iteration, improved_time_iteration

import dolang
from dolang.language import eval_data
from dolang.factory import FlatFunctionFactory
from dolang.symbolic import sanitize, parse_string, str_expression


class AggregateException(Exception):
    pass


class HModel:
    def __init__(self, fname, i_options={}, dptype=None, debug=False):

        txt = read_file_or_url(fname)

        try:
            model_data, hmodel_data = yaml.compose_all(txt, Loader=yaml.BaseLoader)
        except Exception as ex:
            print(
                "Error while parsing YAML file. Probable YAML syntax error in file : ",
                fname,
            )
            raise ex

        self.__model__ = Model(model_data)
        self.data = hmodel_data

        self.discretization_options = i_options

        # cache for functions
        self.__symbols__ = None
        self.__equilibrium__ = None
        self.__projection__ = None
        self.__features__ = None

        self.debug = debug

        self.check()
        self.__set_changed__()

        from dolo.numeric.processes import IIDProcess, ProductProcess

        # if (
        #     dptype is None
        #     and isinstance(self.model.exogenous, ProductProcess)
        #     and (self.model.exogenous.processes[1], IIDProcess)
        # ):
        #     dptype = "iid"
        # else:
        #     dptype = "mc"
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

    def get_calibration(self):
        
        from dolang.symbolic import remove_timing

        import copy

        calibration = dict()
        for k,v in self.data.get("calibration", {}).items():
            if v.tag=='tag:yaml.org,2002:str':

                expr = parse_string(v)
                expr = remove_timing(expr)
                expr = str_expression(expr)
            else:
                expr = float(v.value)
            kk = remove_timing(parse_string(k))
            kk = str_expression(kk)

            calibration[kk] = expr

        from dolang.triangular_solver import solve_triangular_system

        return solve_triangular_system(calibration)

    @property
    def features(self):
        if self.__features__ is None:
            __features__ = {}
            __features__["ex-ante-identical"] = not ("distribution" in self.data)
            __features__["conditional-processes"] = isinstance(
                self.model.exogenous, Conditional
            )
            __features__["iid-shocks"] = isinstance(
                self.model.exogenous, ProductProcess
            ) and (
                False
                not in [
                    isinstance(e, IIDProcess)
                    for e in self.model.exogenous.processes[1:]
                ]
            )
            self.__features__ = __features__
        return self.__features__

    @property
    def name(self):
        return self.data.get("name", "unnamed")

    def check(self):

        kp = self.data.get("projection", None)
        if kp is None:
            raise AggregateException("Missing 'projection section'.")

        # idiosyms = self.model.symbols['exogenous']
        # print(kp.value)
        # vals = [*kp.keys()]

        # targets = [e for e in idiosyms if e in vals]
        # if len(targets) < len(vals):
        #     diff = set(vals).difference(set(targets))
        #     raise AggregateException(
        #         f"Some projected values were not defined as exogenous in the agent's program: {', '.join(diff)}"
        #     )
        # expected = idiosyms[:len(targets)]
        # if tuple(targets) != tuple(expected):
        #     raise AggregateException(
        #         f"Projected values must match first exogenous variables of model. Found {', '.join(targets)}. Expected {', '.join(expected)}"
        #     )

    def set_calibration(self, *pargs, **kwargs):
        if len(pargs) == 1:
            self.set_calibration(**pargs[0])
        self.__set_changed__()
        self.data["calibration"].update(kwargs)

    @property
    def symbols(self):
        if self.__symbols__ is None:
            from dolo.compiler.misc import LoosyDict, equivalent_symbols
            from dolang.symbolic import remove_timing, parse_string, str_expression

            symbols = LoosyDict(equivalences=equivalent_symbols)
            for sg in self.data['symbols'].keys():
                symbols[sg] =  [s.value for s in self.data['symbols'][sg]]

            self.__symbols__ = symbols

        return self.__symbols__     


    @property
    def variables(self):
        return sum([self.symbols[v] for v in ["exogenous", "aggregate"]], [])

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = self.get_calibration()
            from dolo.compiler.misc import CalibrationDict, calibration_to_vector
            calib = calibration_to_vector(self.symbols, calibration_dict)
            self.__calibration__ = CalibrationDict(self.symbols, calib)  #
        return self.__calibration__

    @property
    def exogenous(self):

        # old style only
        calib = self.calibration.flat
        return eval_data(self.data["exogenous"], calibration=calib)

    @property
    def distribution(self):

        # old style only
        calib = self.calibration.flat
        if "distribution" in self.data:
            return eval_data(self.data["distribution"], calibration=calib)
        else:
            return None

    @property
    def projection(self):  # , m: 'n_e', y: "n_y", p: "n_p"):

        if self.__projection__ is None:

            arguments_ = {
                "m": [(e, 0) for e in self.symbols["exogenous"]],
                "y": [(e, 0) for e in self.symbols["aggregate"]],
                "p": self.symbols["parameters"],
            }

            vars = sum([[e[0] for e in h] for h in [*arguments_.values()][:-1]], [])

            arguments = {
                k: [dolang.symbolic.stringify_symbol(e) for e in v]
                for k, v in arguments_.items()
            }

            preamble = {}  # for now
            
            from dolang.symbolic import sanitize, stringify
            eqs = parse_string(self.data['projection'], start='assignment_block')
            eqs = sanitize(eqs, variables=vars) # just to replace (v,) by (v,0) # TODO: remove
            eqs = stringify(eqs)

            content = {}
            for eq in eqs.children:
                lhs, rhs = eq.children
                content[str_expression(lhs)] = str_expression(rhs)

            fff = FlatFunctionFactory(preamble, content, arguments, "equilibrium")
            _, gufun = dolang.function_compiler.make_method_from_factory(
                fff, debug=self.debug
            )

            from dolang.vectorize import standard_function

            self.__projection__ = standard_function(gufun, len(content))

        return self.__projection__

    @property
    def ℰ(self):

        if self.__equilibrium__ is None:

            arguments_ = {
                "e": [(e, 0) for e in self.model.symbols["exogenous"]],
                "s": [(e, 0) for e in self.model.symbols["states"]],
                "x": [(e, 0) for e in self.model.symbols["controls"]],
                "m": [(e, 0) for e in self.symbols["exogenous"]],
                "y": [(e, 0) for e in self.symbols["aggregate"]],
                "p": self.symbols["parameters"],
            }

            vars = sum([[e[0] for e in h] for h in [*arguments_.values()][:-1]], [])

            arguments = {
                k: [dolang.symbolic.stringify_symbol(e) for e in v]
                for k, v in arguments_.items()
            }

            preamble = {}  # for now

            from dolang.symbolic import sanitize, stringify

            eqs = parse_string(self.data['equilibrium'], start="equation_block")
            eqs = sanitize(eqs, variables=vars) 
            eqs = stringify(eqs)
            content = {}
            for i, eq in enumerate(eqs.children):
                lhs, rhs = eq.children
                content[f"eq_{i}"] = "({1})-({0})".format(str_expression(lhs),str_expression(rhs))

            fff = FlatFunctionFactory(preamble, content, arguments, "equilibrium")
            _, gufun = dolang.function_compiler.make_method_from_factory(
                fff, debug=self.debug
            )
            from dolang.vectorize import standard_function

            self.__equilibrium__ = standard_function(gufun, len(content))

        return self.__equilibrium__

    def τ(self, m, p):
        # exogenous process is assumed to be deterministic
        # TEMP:  works only if exogenous is an AR1
        ρ = self.exogenous.ρ
        return ρ * m

    def 𝒜(self, grids, m0: "n_e", μ0: "n_m.N", xx0: "n_m.N.n_x", y0: "n_y", p: "n_p"):

        from dolo.numeric.processes import EmptyGrid
        import numpy as np

        μ0 = np.array(μ0)
        ℰ = self.ℰ
        exg, eng = grids
        if isinstance(exg, EmptyGrid):
            # this is so sad
            mi = self.model.calibration["exogenous"][None, :]  # not used anyway...
        else:
            mi = exg.nodes
        s = eng.nodes
        res = sum(
            [
                μ0[i, :] @ ℰ(mi[i, :], s, xx0[i, :, :], m0, y0, p)
                for i in range(xx0.shape[0])
            ]
        )
        return res

    def get_starting_rule(self, method="improved_time_iteration", **kwargs):
        # provides initial guess for d.r. by solving agent's problem

        dp = self.model.exogenous.discretize()
        # dr0 = time_iteration(self.model, dprocess=mc)
        if method == "improved_time_iteration":
            sol = improved_time_iteration(self.model, dprocess=dp, **kwargs)
        elif method == "time_iteration":
            sol = time_iteration(self.model, dprocess=dp, **kwargs)
        dr0 = sol.dr

        return dr0
