# -*- coding: utf-8 -*-
# %%
from dolo import *
from dolark.perturbation import AggregateModel
from matplotlib import pyplot as plt

# %% [markdown]
# First we check that we can solve the one-agent model

# %%
discretization_options = {"N": 2}
model = yaml_import('ayiagari.yaml')
mc = model.exogenous.discretize(to='mc', options=[{},discretization_options])
sol0 = time_iteration(model, details=True, dprocess=mc)


# %% [markdown]
# Now we set set aggregate conditions. For now, this is done with a Python class.
# Later, this information will be contained, in a specialized yaml file.

# %%
class KrussellSmith(AggregateModel):

    symbols = dict(
        exogenous = ["z"],
        aggregate = ["K"],
        parameters = ["A", "alpha", "delta", 'ρ']
    )

    calibration_dict = dict(
        A = 1,
        alpha = 0.36,
        delta = 0.025,
        K = 40,
        z = 0,
        ρ = 0.95
    )

    def τ(self, m, p):
        # exogenous process is assumed to be deterministic
        ρ = p[3]
        return m*ρ

    def definitions(self, m: 'n_e', y: "n_y", p: "n_p"):
        from numpy import exp
        z = m[0]
        K = y[0]
        A = [0]
        alpha = p[1]
        delta = p[2]
        N = 1
        r = alpha*exp(z)*(N/K)**(1-alpha) - delta
        w = (1-alpha)*exp(z)*(K/N)**(alpha)
        return {'r': r, "w": w}

    def 𝒜(self, m0: 'n_e', μ0: "n_m.N" , xx0: "n_m.N.n_x", y0: "n_y", p: "n_p"):

        import numpy as np
        kd = sum( [float((μ0[i,:]*xx0[i,:,0]).sum()) for i in range(μ0.shape[0])] )
        aggres_0 = np.array( [kd - y0[0] ])
        return aggres_0

# %%
# We create an aggregate model
aggmodel = KrussellSmith(model, sol0.dr)
aggmodel # TODO: find a reasonable representation of this object

# %%

# %%
# We can now solve for the aggregate equilibrium
eq = aggmodel.find_steady_state()
eq

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
# lot's look at the aggregate equilibrium
for i in range(eq.μ.shape[0]):
    s = eq.dr.endo_grid.nodes() # grid for states (temporary)
    plt.plot(s, eq.μ[i,:]*(eq.μ[i,:].sum()), label=f"y={eq.dr.exo_grid.node(i)[2]: .2f}")
plt.plot(s, eq.μ.sum(axis=0), label='total', color='black')
plt.grid()
plt.legend(loc='upper right')
plt.title("Wealth Distribution by Income")

# %%
# alternative way to plot equilibrium
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
peq = aggmodel.perturb(eq)


# %%
# and we simulate given initial value of aggregate shock
sim = peq.response([0.1])

# %%
plt.subplot(121)
for t, (m,μ,x,y) in enumerate(sim):
    plt.plot(μ.sum(axis=0), color='red', alpha=0.01)
plt.xlabel('a')
plt.ylabel('density')
plt.subplot(122)
plt.plot( [e[3][0] for e in sim])
plt.xlabel("t")
plt.ylabel("k")
plt.tight_layout()

# %%
# Let's check the effect of the number of discretization points for the exogenous processresults = []
results = []
for N in range(2,6):
    print(f"--- Computing model with N={N} ---")
    # discretization_options is passed to the discretize method for idiosyncartic shocks
    aggmodel = KrussellSmith(model,  discretization_options={"N":N})
    eq = aggmodel.find_steady_state()
    peq = aggmodel.perturb(eq)
    results.append((aggmodel, eq, peq))


# %%
# Let's see how discretization affects the total distribution
for i,(aggmodel, eq, pert) in enumerate(results):
    N = i+2
    plt.plot( eq.μ.sum(axis=0), label=f"N={N}")
plt.legend(loc='upper right')

# %%

# %%
