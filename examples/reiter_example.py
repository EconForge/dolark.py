# -*- coding: utf-8 -*-
# %%
from dolo import groot
groot('examples')

# %%
from matplotlib import pyplot as plt

# %%
# here are the three functions we use from dolark
from dolark import HModel
from dolark.equilibrium import find_steady_state
from dolark.perturbation import perturb

# %%
# Let's import the heterogeneous agents model
aggmodel = HModel('ayiagari.yaml')
aggmodel # TODO: find a reasonable representation of this object


# %%
# see what can be done
aggmodel.features

# %%

# %% [markdown]
# First we can check whether the one-agent sub-part of it works, or whether we will need to find better initial guess.

# %%
from dolo import time_iteration
i_opts = {"N": 2} # discretization options, consistent, with current implementation of aggregate perturbation
model = aggmodel.model
mc = model.exogenous.discretize(to='mc', options=[{},i_opts])
sol0 = time_iteration(model, details=True, dprocess=mc)

# %%
# We can now solve for the aggregate equilibrium
eq = find_steady_state(aggmodel)
eq

# %%
# lot's look at the aggregate equilibrium
for i in range(eq.μ.shape[0]):
    s = eq.dr.endo_grid.nodes # grid for states (temporary)
    plt.plot(s, eq.μ[i,:]*(eq.μ[i,:].sum()), label=f"y={eq.dr.exo_grid.node(i)[2]: .2f}")
plt.plot(s, eq.μ.sum(axis=0), label='total', color='black')
plt.grid()
plt.legend(loc='upper right')
plt.title("Wealth Distribution by Income")

# %%
# alternative way to plot equilibrium

import altair as alt
df = eq.as_df()
spec = alt.Chart(df).mark_line().encode(
    x = 'a',
    y = 'μ',
    color = 'i_m:N'
)
spec

# %%
# alternative way to plot equilibrium (with some interactivity)
# TODO: function to generate it automatically.

import altair as alt
single = alt.selection_single(on='mouseover', nearest=True)
df = eq.as_df()
ch = alt.Chart(df)
spec = ch.properties(title='Distribution', height=100).mark_line().encode(
    x = 'a',
    y = 'μ',
    color = alt.condition(single, 'i_m:N', alt.value('lightgray'))
).add_selection(
        single
) + ch.mark_line(color='black').encode(
    x = 'a',
    y = 'sum(μ)'
) & ch.properties(title='Decision Rule', height=100).mark_line().encode(
    x = 'a',
    y = 'i',
    color = alt.condition(single, 'i_m:N', alt.value('lightgray'))
).add_selection(
        single
)

# %%

# Resulting object can be saved to a file. (try to open this file in jupyterlab)
open('distrib.json','tw').write(spec.to_json())

# %%

# %%
import xarray

# %%
# now we compute the perturbation
peq = perturb(aggmodel, eq)


# %%
# and we simulate given initial value of aggregate shock
sim = peq.response([0.1])

# %%
plt.subplot(121)
for t, (m,μ,x,y) in enumerate(sim):
    plt.plot(μ.sum(axis=0), color='red', alpha=0.01)
plt.xlabel('a')
plt.ylabel('density')
plt.grid()
plt.subplot(122)
plt.plot( [e[3][0] for e in sim])
plt.xlabel("t")
plt.ylabel("k")
plt.grid()
plt.tight_layout()

# %%
