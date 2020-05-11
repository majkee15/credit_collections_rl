import pytest
from gym import spaces
from space_wrappers.misc import *
try:
    from unittest import mock
except ImportError:
    import mock


@pytest.fixture()
def env():
    yield mock.Mock(autospec=gym.Env)


def test_scalar_action_wrapper(env):
    wrapped = ToScalarActionWrapper(env)
    assert wrapped._action(np.array([1])) == 1
    # TODO do we want an error message here.
    assert wrapped._action([1, 2, 3]) == [1, 2, 3]
    assert list(wrapped._action(np.array([1, 2, 3]))) == [1, 2, 3]


def test_stack_observation(env):
    env.observation_space = spaces.Box(0.0, 1.0, shape=(2, 3), dtype=np.float32)
    env.reset.return_value = np.zeros((2, 3))
    env.step = lambda x: (x, None, None, None)

    wrapped = StackObservationWrapper(env, 5, axis=-1)
    assert wrapped.observation_space == spaces.Box(0.0, 1.0, shape=(2, 3, 5), dtype=np.float32)

    assert wrapped.reset() == pytest.approx(np.zeros((2, 3, 5)))
    o, _, _, _ = wrapped.step(np.ones((2, 3)) * 0.5)
    assert o == pytest.approx(np.concatenate((np.zeros((2, 3, 4)), np.ones((2, 3, 1)) * 0.5), axis=2))


def test_repeat_action_wrapper(env):
    wrapped = RepeatActionWrapper(env, 3)
    env.step.return_value = (5, 1.0, False, {})
    obs, rew, done, info = wrapped.step(3)

    assert env.step.call_count == 4

    assert obs == 5
    assert rew == 4
    assert done is False
    assert info == {'skip.stepcount': 4}

