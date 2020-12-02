import numpy as np
from dolo.numeric.processes import ProductProcess, ConstantProcess, IIDProcess


def discretize_idiosyncratic_shocks(distributions, options=None):
    assert len(distributions) == 1
    if options is None:
        options = [{}] * len(distributions)
    disdist = []
    for i, (k, v) in enumerate(distributions.items()):
        for x, y in v.discretize(to="iid", **options[i]).iteritems(0):
            disdist.append((x, {k: float(y[0])}))
    return disdist


def inject_process(process, exogenous, to=None, options={}):

    if isinstance(process, np.ndarray) and isinstance(exogenous, ProductProcess):

        q0 = process

        assert process.ndim == 1
        if to is None:
            if isinstance(exogenous.processes[1], IIDProcess):
                to = "iid"
            else:
                to = "mc"

        exg = ProductProcess(ConstantProcess(μ=q0), *exogenous.processes[1:])
    else:
        raise Exception("Not implemented.")

    dp = exg.discretize(to=to, options=options)

    return dp
