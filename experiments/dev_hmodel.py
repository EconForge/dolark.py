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

for i, (w, eq) in enumerate(eqss):
    s =eq.dr.endo_grid.nodes()
    for j in range(3):
        plt.plot(s, w*eq.μ[j,:], label=f"{i}" )
plt.legend()

# wealth distribution
bins = []
for i, (w, eq) in enumerate(eqss):
    bins.append( w*sum(s.ravel()*eq.μ.sum(axis=0)))


plt.plot([e[1]['β'] for e in dist], bins, '-o')
plt.xlabel('β')

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
#%%

from dolark.perturbation import perturb
from dolo import groot
from dolark import HModel
from dolark.equilibrium import find_steady_state

groot("examples")

hmodel1 = HModel('ayiagari.yaml')
print(hmodel1.name)
eq = find_steady_state(hmodel1)
perteq = perturb(hmodel1, eq)
