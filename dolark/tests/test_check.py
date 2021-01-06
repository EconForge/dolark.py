# %%
import pytest

# from dolo import groot
# groot("dolark/tests/check")
from dolark import HModel
from dolark.model import AggregateException

# %%
def state_None():
    HModel("./dolark/tests/check/error_state_None.yaml")


def trans_None():
    HModel("error_trans_None.yaml")


def state_trans_1():
    HModel("error_state_trans_1.yaml")


def state_trans_2():
    HModel("error_state_trans_2.yaml")


# %%
# def test_state_None():
#     with pytest.raises(AggregateException):
#         state_None()

# def test_trans_None():
#     with pytest.raises(AggregateException):
#         trans_None()

# def test_state_trans_1():
#     with pytest.raises(AggregateException):
#         state_trans_1()

# def test_state_trans_2():
#     with pytest.raises(AggregateException):
#         state_trans_2()

# %%
