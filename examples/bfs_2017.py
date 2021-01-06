# %%

# WARNING: this is not working yet

# +
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



#%%

mix = hmodel.agent.__exogenous__.processes[2]
nn = mix.distributions['1']

sum( [w * n for w,n in nn.discretize().items()] )

hmodel.agent.calibration['τ']




# %%

# Agent's distributions look good

# dis_iids = []
# for i in range(3):
#     dis_iids.append(hmodel.agent.exogenous.processes[i].discretize(to='iid'))


# df1 = pd.DataFrame([[w,x[0]] for w,x in dis_iids[0].iteritems(0)], columns=['w', 'x'])  #constant
# df2 = pd.DataFrame([[w,x[0]] for w,x in dis_iids[1].iteritems(0)], columns=['w', 'x'])
# df3 = pd.DataFrame([[w,x[0]] for w,x in dis_iids[2].iteritems(0)], columns=['w', 'x'])


# alt.Chart(df2, title="Ψ").mark_bar().encode(x="x", y="w") & alt.Chart(
#     df3, title="Ξ"
# ).mark_bar().encode(x="x", y="w")

# Agents' decision rule is not correct yet

from dolo import time_iteration, improved_time_iteration

hmodel.agent.__exogenous__.processes


mc_0 = hmodel.agent.__exogenous__.processes[0].discretize(to='mc')
mc_1 = hmodel.agent.__exogenous__.processes[1].discretize(to='mc')
mx = hmodel.agent.__exogenous__.processes[2].discretize(to='mc')

hmodel.agent.__exogenous__.processes[0]
hmodel.agent.__exogenous__.processes[1]
hmodel.agent.__exogenous__.processes[2]

grid, dprocess = hmodel.agent.discretize()

dr = time_iteration(hmodel.agent)
# # %time dr = improved_time_iteration(hmodel.model, dr0=dr, verbose=True, details=False)

# %%

from matplotlib import pyplot as plt
from dolo import tabulate

tab = tabulate(hmodel.agent, dr, "m")

plt.plot(tab["m"], tab["c"])

# %%

plt.plot(pps_normal, pps_normal*0,'o')
plt.plot(pps_lognormal, pps_lognormal*0, 'x')

# %%


# ergodic distribution (premature)

Π, μ = ergodic_distribution(hmodel.model, dr)
df_μ = μ.to_dataframe("μ").reset_index()

ch = alt.Chart(tab)
g1 = ch.mark_line(color="black", strokeDash=[1, 1]).encode(
    x="m", y="m"
) + ch.mark_line().encode(x="m", y="c")
g2 = alt.Chart(df_μ).mark_line().encode(x="m:Q", y="mu:Q")

g2

# There seem to be something wrong with the calibration at the aggregate level

# here are the value of i.r. and w calibrated at the agent's level
hmodel.agent.calibration["r", "w"]

# here are the values projected from market equilibrium, given default level of capital
m0, y0, p = hmodel.calibration["exogenous", "aggregate", "parameters"]
hmodel.projection(m0, y0, p)  # values for r, w, ω (not the same at all)

# %%
