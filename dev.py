# %%
from dolo import groot

groot("examples")
from dolark import HModel
from dolark.equilibrium import find_steady_state

hmodel = HModel("ayiagari_betadist.yaml")
eq = find_steady_state(hmodel)