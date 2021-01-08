# %%

from dolo import *

# from matplotlib import pyplot as plt
import pandas as pd
import altair as alt
from dolark import HModel

groot("examples")

hmodel = HModel("bfs_2017.yaml")
hmodel.features
# -
hmodel.agent


#%%

x1 = hmodel.agent.eval_formula("(D+r)/Đ")
x2 = hmodel.agent.eval_formula("β")

print(f"Effective discount rate (D+r)/Đ = {x1}")
print(f"Time discount {x2}")


#%%

dr = time_iteration(hmodel.agent, maxit=100)
sol = improved_time_iteration(hmodel.agent, dr0=dr, verbose=True)
dr = sol.dr

#%%

from dolo.algos.ergodic import ergodic_distribution
μ = ergodic_distribution(hmodel.agent, dr)[1]

#%%

from matplotlib import pyplot as plt
plt.plot(μ.data.ravel())


#%%

from dolark.equilibrium import find_steady_state
eqs = find_steady_state(hmodel, dr0 =dr, verbose='full', return_fun=False)


#%%

m0 = hmodel.calibration['exogenous']
y0 = hmodel.calibration['aggregate']
p0 = hmodel.calibration['parameters']

(m0,y0, hmodel.projection(m0,y0,p0))


# %%

from matplotlib import pyplot as plt
from dolo import tabulate

tab = tabulate(hmodel.agent, dr, "m")

plt.plot(tab["m"], tab["m"])
plt.plot(tab["m"], tab["c"])
plt.grid(True)

# %%



from dolark.perturbation import perturb
sol = perturb(hmodel, eqs)
# %%
