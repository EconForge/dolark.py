# -*- coding: utf-8 -*-
# + {}
# This notebook computes the equilibrium distribution for various types of models
# -

from dolark import HModel
from dolark.equilibrium import find_steady_state

# +
hmodel1 = HModel('ayiagari.yaml')
print(hmodel1.name)

print(hmodel1.features)
# -

hmodel2 = HModel('ayiagari_betadist.yaml')
print(hmodel2.name)
print(hmodel2.features)
print(hmodel2.distribution)

hmodel3 = HModel('bfs_2017.yaml')
print(hmodel3.name)
print(hmodel3.features)
# print(hmodel3.distribution)

# # Identical Agents: autocorrelated procesess

# +
# the agent's problem has autocorrelated exogenous process
# it is discretized as an markov chain
# -

eq = find_steady_state(hmodel1)
eq

# a bit out of topic here:
from dolark.perturbation import perturb
peq = perturb(hmodel1, eq)
peq

# +
# cf reiter_example to see what to do with eq and peq
# -

# # Many Agents: autocorrelated procesess

# distribution of agent's parameters
hmodel2.distribution

from dolark.shocks import discretize_idiosyncratic_shocks
dist = discretize_idiosyncratic_shocks(hmodel2.distribution, options=[{'N':6}])
dist

eqs = find_steady_state(hmodel2)
eqs # results is (for now) a list of equilibrium objects

s = eqs[0][1].dr.endo_grid.nodes().ravel()
for (w,eq) in eqs:
    dens = eq.μ.sum(axis=0) # \mu is a 2d array: exogenous x endogenous states
    # we sum all agents of a given type across the exogenous dimension
    plt.plot(s, dens, label=f"β={dist[i][1]['β']:.3f}")
plt.grid()

# Oups, the more patient agents are diverging...

# ### we can do the same by hand

from tqdm import tqdm_notebook as tqdm
import numpy as np
from dolo import time_iteration

dp = hmodel2.model.exogenous.discretize(to='mc')
dr0 = time_iteration(hmodel2.model, dprocess=dp, verbose=False)

from dolark.equilibrium import equilibrium as equilibrium_fun
m0 = hmodel2.calibration['exogenous']
kvec = np.linspace(30, 40, 20)
resids = []
# we compute market residual by agent
for w, kwargs in tqdm(dist):
    hmodel2.model.set_calibration(**kwargs)
    res = [equilibrium_fun(hmodel2, m0, np.array([k]), dr0=dr0, return_equilibrium=False) for k in kvec]
    resids.append(res)

# +
resids = [np.array(e).ravel() for e in eqs]
from matplotlib import pyplot as plt
Na = len(dist)
for i,eq in enumerate(equils):
    plt.plot(kvec, kvec-eq, label=f"β={dist[i][1]['β']:.3f}")
plt.plot(kvec, kvec-sum(resids,0)/Na, linestyle='--', color='black', label='Total Demand')
plt.plot(kvec, kvec, color='black', label='Total Supply')
plt.legend(loc='upper right')

plt.grid()
# -

# # Many agents: iid shocks

# hmodel3.get_starting_rule is broken here
# we need to supply an initial rule here
dp = hmodel3.model.exogenous.discretize(to='iid')
dr0 = time_iteration(hmodel3.model, verbose=False)

eqs = find_steady_state(hmodel3, dr0=dr0)

# +
# doesn't work so far
