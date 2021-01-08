#%%

from dolark import HModel
from dolo.algos import time_iteration, improved_time_iteration


hmodel = HModel("bfs_2017_K.yaml")
hmodel.features

#%%

#%%

m0 = hmodel.calibration['exogenous']
s0 = hmodel.calibration['states']
y0 = hmodel.calibration['aggregate']
p0 = hmodel.calibration['parameters']

(m0,y0, hmodel.projection(m0,s0,y0,p0))


# %%

dr = time_iteration(hmodel.agent, maxit=100, verbose=True)
sol = improved_time_iteration(hmodel.agent, dr0=dr, verbose=True)
dr = sol.dr

# %%

from dolo.algos.ergodic import ergodic_distribution
μ = ergodic_distribution(hmodel.agent, dr)[1]


from matplotlib import pyplot as plt
plt.plot(μ.data.ravel())



# %%

from dolark.equilibrium import find_steady_state
eq = find_steady_state(hmodel, verbose=True, dr0=dr)

# %%

from matplotlib import pyplot as plt
plt.plot( eq.μ )



# %%

# %%
