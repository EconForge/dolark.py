# %%
from dolo import groot

groot("examples")
from dolark import HModel
from dolark.equilibrium import find_steady_state

# %%
def test_steady_state_non_ex_ante_ha():
    hmodel = HModel("ayiagari.yaml")
    eq = find_steady_state(hmodel)
    hmodel = HModel("bfs_2017.yaml")
    eq = find_steady_state(hmodel)


def test_steady_state_ex_ante():
    hmodel = HModel("ayiagari_betadist.yaml")
    eq = find_steady_state(hmodel)


def test_with_agg_states():
    hmodel = HModel("prototype.yaml")
    eq = find_steady_state(hmodel)
