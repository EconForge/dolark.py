# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from dolo import *
from matplotlib import pyplot as plt
import pandas as pd
import altair as alt

model = yaml_import("bfs_2017.yaml")

model.exogenous.processes[0].discretize(to='iid')
model.exogenous.processes[1].discretize(to='iid')
iim = model.exogenous.processes[2].discretize(to='iid')

df = pd.DataFrame([[w,x[0]] for w,x in iim.iteritems(0)], columns=['w', 'x'])

alt.Chart(df).mark_bar().encode(x='x', y='w')
# -
from dolo import time_iteration, improved_time_iteration
dr = time_iteration(model)
# # %time dr = improved_time_iteration(model)


from dolo import tabulate

tab = tabulate(model, dr, 'm')
Π, μ = ergodic_distribution(model, dr)
df_μ = μ.to_dataframe('μ').reset_index()



ch = alt.Chart(tab)
g1 = ch.mark_line(color='black',strokeDash=[1,1]).encode(x='m', y='m') + \
ch.mark_line().encode(x='m', y='c')
g2 = alt.Chart(df_μ).mark_line().encode(x='m:Q', y= 'mu:Q')

g2

plt.plot(df_μ['m'], df_μ['μ'])

df_μ.dtypes


