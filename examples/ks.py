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

# WARNING: this is not working yet
from dolark import HModel
from dolo import pcat

hmodel = HModel("ks.yaml")

# The yaml file allows for the agent's exogenous process to depend on the aggregate values.

# !sed -n 40,51p ks.yaml | pygmentize -l yaml

# Here is what is recovered from the yaml file

exo = hmodel.model.exogenous

exo.condition  # the driving process coming from aggregate simulation

μ = exo.condition.μ
μ

exo.arguments(μ)

exo.arguments(μ + 0.1)

from dolo.numeric.processes import MarkovChain

# here is how one can contruct a markov chain
mc = MarkovChain(**exo.arguments(μ + 0.1))
