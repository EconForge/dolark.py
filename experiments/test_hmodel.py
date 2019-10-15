import numpy as np
import copy
import scipy
from dolark.equilibrium import equilibrium, find_steady_state
from dolo import improved_time_iteration, ergodic_distribution, time_iteration
from dolo import time_iteration, improved_time_iteration
from dolo import groot
groot('examples')

# Let's import the heterogeneous agents model
from dolark import HModel

hmodel1 = HModel('ayiagari.yaml')
print(hmodel1.name)

hmodel2 = HModel('ayiagari_betadist.yaml')
print(hmodel2.name)

# hmodel3 = HModel('bfs_2017.yaml')
# print(hmodel3.name)

#%%


#%%



#%%

# dr0 = hmodel2.get_starting_rule()
# m0, y0 = hmodel2.calibration['exogenous','aggregate']
# eq = equilibrium(hmodel2, m0, y0, dr0=dr0)
# eq = find_steady_state(hmodel1, dr0=dr0)
# #%%

eq0 = find_steady_state(hmodel1)

for j in range(3):
    plt.plot( w*eq0.μ[j,:], label=f"{i}" )

#%%

eqss = find_steady_state(hmodel2)

from matplotlib import pyplot as plt
plt.plot( eq.μ[0,:] )

for i, (w, eq) in enumerate(eqss):
    for j in range(3):
        plt.plot( w*eq.μ[j,:], label=f"{i}" )
y0 = eqss[0][1].y

#%%

# by hand

from dolark.shocks import discretize_idiosyncratic_shocks

dist = discretize_idiosyncratic_shocks(hmodel2.distribution)

from tqdm import tqdm


dr0 = hmodel2.get_starting_rule()
m0 = hmodel2.calibration['exogenous']

kvec = np.linspace(20, 40, 20)
eqs = []
for w, kwargs in tqdm(dist):
    hmodel2.model.set_calibration(**kwargs)
    res = [equilibrium(hmodel2, m0, np.array([k]), dr0=dr0, return_equilibrium=False) for k in kvec]
    eqs.append(res)

dist

#%%

eqs = [np.array(e).ravel() for e in eqs]
from matplotlib import pyplot as plt

for eq in eqs:
    plt.plot(kvec, kvec-eq)
plt.plot(kvec, kvec-sum(eqs,0)*0.5, linestyle='--', color='black')
plt.plot(kvec, kvec, color='black')

plt.grid()

eq0.y

# #%%

# m0, y0 = hmodel1.calibration['exogenous','aggregate']
# equilibrium(hmodel1, m0, y0)

# #%%

# m0, y0 = hmodel2.calibration['exogenous','aggregate']
# equilibrium(hmodel2, m0, y0)


# #%%
# m0, y0, p0 = hmodel3.calibration['exogenous','aggregate','parameters']

# q0 = hmodel3.projection(m0,y0,p0)

# dr3 = time_iteration(hmodel3.model, verbose=False)

# #%%
# y0 = np.array([8.0])
# %time res, (sol, μ0, Π0) = equilibrium(hmodel3, m0, y0, dr0=dr3)




# from dolark.perturbation import Equilibrium, colored
# import scipy


#%%
