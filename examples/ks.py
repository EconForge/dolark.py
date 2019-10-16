# -*- coding: utf-8 -*-
from dolark import HModel
from dolo import pcat

hmodel = HModel('ks.yaml')

# !sed -n 40,51p ks.yaml | pygmentize -l yaml

exo = hmodel.model.exogenous

exo.condition # the driving process coming from aggregate simulation

μ = exo.condition.μ
μ

exo.arguments(μ)

exo.arguments(μ+0.1)

from dolo.numeric.processes import MarkovChain

# here is how one can contruct a markov chain
mc = MarkovChain(**exo.arguments(μ+0.1))
