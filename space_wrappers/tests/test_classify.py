from space_wrappers.classify import *
from gym import Space
from gym.spaces import *
import numpy as np
import pytest


class UnknownSpace(Space):
    pass


# is_discrete
def test_is_discrete():
    assert is_discrete(Discrete(10))
    assert is_discrete(MultiDiscrete([4, 5]))
    assert is_discrete(MultiBinary(5))
    assert is_discrete(Tuple((Discrete(5), Discrete(4))))
    assert not is_discrete(Box(np.zeros(2), np.ones(2), dtype=np.float32))

    with pytest.raises(TypeError):
        is_discrete(5)

    with pytest.raises(NotImplementedError):
        is_discrete(UnknownSpace())


def test_is_compound():
    assert not is_compound(Discrete(10))
    assert is_compound(MultiDiscrete([4, 5]))
    assert is_compound(MultiBinary(5))
    assert is_compound(Tuple((Discrete(5), Discrete(4))))
    assert is_compound(Box(np.zeros(2), np.ones(2), dtype=np.float32))
    assert not is_compound(Box(np.zeros(1), np.ones(1), dtype=np.float32))

    with pytest.raises(TypeError):
        is_compound(5)

    with pytest.raises(NotImplementedError):
        is_compound(UnknownSpace())


def test_is_flat():
    assert is_flat(Discrete(10))
    assert not is_flat(MultiDiscrete([4, 5]))
    assert not is_flat(MultiBinary(5))
    assert not is_flat(Tuple((Discrete(5), Discrete(4))))
    assert is_flat(Box(np.zeros(2), np.ones(2), dtype=np.float32))
    assert not is_flat(Box(np.zeros((2, 3)), np.ones((2, 3)), dtype=np.float32))

    with pytest.raises(TypeError):
        is_flat(5)

    with pytest.raises(NotImplementedError):
        is_flat(UnknownSpace())


def test_num_discrete_actions():
    with pytest.raises(TypeError):
        num_discrete_actions(Box(np.zeros(2), np.ones(2), dtype=np.float32))

    assert num_discrete_actions(Discrete(10)) == (10,)
    assert num_discrete_actions(MultiDiscrete([5, 6])) == (5, 6)
    assert num_discrete_actions(MultiBinary(3)) == (2, 2, 2)

    with pytest.raises(NotImplementedError):
        num_discrete_actions(UnknownSpace())
