# %%
import numpy as np
import copy
import scipy
from dolark.equilibrium import equilibrium, find_steady_state
from dolo import improved_time_iteration, ergodic_distribution, time_iteration
from dolo import time_iteration, improved_time_iteration
from dolo import groot

groot("examples")

from dolark import HModel

# %%
hmodel1 = HModel("ayiagari.yaml")
print(hmodel1.name)

hmodel2 = HModel("ayiagari_betadist.yaml")
print(hmodel2.name)
# %%
eq1 = find_steady_state(hmodel1)
eq2 = find_steady_state(hmodel2)
# %%
# Distribution of assets for each idiosyncratic state
from matplotlib import pyplot as plt

s = eq1.dr.endo_grid.nodes
for j in range(3):
    plt.plot(s, eq1.μ[j, :], label=f"{j}")
plt.legend()
# %%
from matplotlib import pyplot as plt

for i, (w, eq) in enumerate(eq2):
    s = eq.dr.endo_grid.nodes
    for j in range(3):
        plt.plot(s, w * eq.μ[j, :], label=f"{i}")
plt.legend()
# %%
# Wealth distribution
s = eq1.dr.endo_grid.nodes
plt.plot(s, eq1.μ.sum(axis=0), "-o")
# %%
bins = []
for w, eq in eq2:
    bins.append(w * sum(s.ravel() * eq.μ.sum(axis=0)))

from dolark.shocks import discretize_idiosyncratic_shocks

dist = discretize_idiosyncratic_shocks(hmodel2.distribution)
plt.plot([e[1]["β"] for e in dist], bins, "-o")
plt.xlabel("β")
# %%
# Steady-state equilibrium: graphic resolution
dr0 = hmodel1.get_starting_rule()
m0 = hmodel1.calibration["exogenous"]

kvec = np.linspace(30, 50, 20)
hmodel1.model.set_calibration(**kwargs)
eqs = [
    equilibrium(hmodel1, m0, np.array([k]), dr0=dr0, return_equilibrium=False)
    for k in tqdm(kvec)
]

eqs = [e[0] for e in eqs]
# %%
from matplotlib import pyplot as plt

plt.plot(kvec, kvec - eqs, color="black")
# %%
from tqdm import tqdm

dr0 = hmodel2.get_starting_rule()
m0 = hmodel2.calibration["exogenous"]

kvec = np.linspace(20, 40, 20)
eqs = []
for w, kwargs in tqdm(dist):
    hmodel2.model.set_calibration(**kwargs)
    res = [
        equilibrium(hmodel2, m0, np.array([k]), dr0=dr0, return_equilibrium=False)
        for k in kvec
    ]
    eqs.append(res)
#%%
eqs = [np.array(e).ravel() for e in eqs]
from matplotlib import pyplot as plt

for eq in eqs:
    plt.plot(kvec, kvec - eq)
plt.plot(kvec, kvec - sum(eqs, 0) * 0.5, linestyle="--", color="black")
plt.plot(kvec, kvec, color="black")
plt.grid()
#%%

from dolark.perturbation import perturb
from dolo import groot
from dolark import HModel
from dolark.equilibrium import find_steady_state

groot("examples")

hmodel1 = HModel("ayiagari.yaml")
print(hmodel1.name)
eq = find_steady_state(hmodel1)
perteq = perturb(hmodel1, eq1)
# %%
hmodel3 = HModel("bfs_2017.yaml")
print(hmodel3.name)

eq3 = find_steady_state(hmodel3)
