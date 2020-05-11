# transform spaces
from gym import Space, spaces
import numpy as np
import itertools
import numbers
from collections import namedtuple
from .classify import *

Transform = namedtuple('Transform', ['original', 'target', 'convert_to', 'convert_from'])


# small helper functions.
def _identity(x):
    return x


class _Lookup(object):
    def __init__(self, source):
        self._source = source

    def __call__(self, key):
        if isinstance(key, (np.ndarray, list)):
            key = tuple(key)
        return self._source[key]


class _LinearTransform(object):
    def __init__(self, offset, slope, dtype=float):
        self._offset = offset
        self._slope = slope
        self._dtype = dtype

    def __call__(self, x):
        return self._dtype(self._offset + self._slope * float(x))


class _LinearTransformArray(object):
    def __init__(self, offset, slope, shape, dtype=float):
        self._offset = offset
        self._slope = slope
        self._dtype = dtype
        self._shape = shape

    def __call__(self, x):
        return np.reshape(self._offset + self._slope * x, self._shape).astype(self._dtype)


class _FlattenTuple(object):
    def __init__(self, subspace_trafos):
        self._subspaces = subspace_trafos

    def __call__(self, x):
        return np.concatenate([trafo.convert_to(val) for trafo, val in zip(self._subspaces, x)])


class _DecomposeTuple(object):
    def __init__(self, subspace_trafos):
        self._subspaces = subspace_trafos

    def __call__(self, x):
        last_pos = 0
        decomposed = []
        for ss in self._subspaces:
            n = ss.target.low.size
            d = x[last_pos:last_pos+n]
            last_pos += n
            decomposed.append(ss.convert_from(d))
        return tuple(decomposed)


# Discretization 
def discretize(space, steps):
    """
    Creates a discretized version of `space` and returns
    a `Transform` that contains the conversion functions.
    If the space is already discrete, the identity
    is returned. The steps are distributed such that the old
    minimum and maximum value can still be reached in the new
    domain.
    :param gym.Space space: The space to be discretized.
    :param int|Iterable steps: The number of discrete steps to produce
                  for each continuous dimension. Can be an
                  Integer or a list.
    :raises ValueError: If less than two steps are are supplied.
    :return Transform: A `Transform` to the discretized space.
    """

    # there are two possible ways how we could handle already
    # discrete spaces. 
    #  1) throw an error because (unless
    #     steps is configured to fit) we would try to convert 
    #     an already discrete space to one with a different number
    #     of states.
    #  2) keep the space as is.
    # here, we implement the second. This allows scripts that 
    # train a discrete agent to just apply discretize, only 
    # changing envs that are not already discrete.

    if is_discrete(space):
        return Transform(space, space, _identity, _identity)

    # check that step number is valid and convert steps into a np array
    if not isinstance(steps, numbers.Integral):
        steps = np.array(steps, dtype=int)
        if (steps < 2).any():
            raise ValueError("Need at least two steps to discretize, got {}".format(steps))
    elif steps < 2:
        raise ValueError("Need at least two steps to discretize, got {}".format(steps))

    if isinstance(space, spaces.Box):
        if len(space.shape) == 1 and space.shape[0] == 1:
            discrete_space = spaces.Discrete(steps)
            lo = space.low[0]
            hi = space.high[0]

            convert = _LinearTransform(lo, (hi-lo) / (steps - 1.0))
            back = _LinearTransform(-lo * (steps-1) / (hi - lo), (steps - 1.0) / (hi-lo), int)
            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=back)
        else:
            if isinstance(steps, numbers.Integral):
                steps = np.full(space.low.shape, steps)
            if steps.shape != space.shape:
                raise ValueError("Supplied steps {} have invalid shape, expected {}".format(steps, steps.shape,
                                                                                            space.shape))

            discrete_space = spaces.MultiDiscrete(steps.flatten())
            lo = space.low.flatten()
            hi = space.high.flatten()

            convert = _LinearTransformArray(lo, (hi - lo) / (steps - 1.0), space.shape)
            back = _LinearTransformArray(-lo * (steps - 1) / (hi - lo), (steps - 1.0) / (hi - lo), (len(steps),), int)

            return Transform(original=space, target=discrete_space, convert_from=convert, convert_to=back)

    raise NotImplementedError("Unknown space {} of type {} supplied".format(space, type(space)))  # pragma: no cover


# Flattening
def flatten(space):
    """
    Flattens a space, which means that for continuous spaces (Box)
    the space is reshaped to be of rank 1, and for multidimensional
    discrete spaces a single discrete action with an increased number
    of possible values is created.
    Please be aware that the latter can be potentially pathological in case
    the input space has many discrete actions, as the number of single discrete
    actions increases exponentially ("curse of dimensionality").
    :param gym.Space space: The space that will be flattened
    :return Transform: A transform object describing the transformation
            to the flattened space.
    :raises TypeError, if `space` is not a `gym.Space`.
            NotImplementedError, if the supplied space is neither `Box` nor
            `MultiDiscrete` or `MultiBinary`, and not recognized as
            an already flat space by `is_compound`.
    """
    # no need to do anything if already flat
    if is_flat(space):
        return Transform(space, space, _identity, _identity)

    if isinstance(space, spaces.Box):
        shape = space.low.shape
        lo = space.low.flatten()
        hi = space.high.flatten()

        def convert(x):
            return np.reshape(x, shape)

        def back(x):
            return np.reshape(x, lo.shape)

        flat_space = spaces.Box(low=lo, high=hi, dtype=space.dtype)
        return Transform(original=space, target=flat_space, convert_from=convert, convert_to=back)

    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        if isinstance(space, spaces.MultiDiscrete):
            ranges = [range(0, k, 1) for k in space.nvec]
        elif isinstance(space, spaces.MultiBinary):  # pragma: no branch
            ranges = [range(0, 2) for i in range(space.n)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        inverse_lookup = {value: key for (key, value) in enumerate(lookup)}
        flat_space = spaces.Discrete(len(lookup))
        return Transform(original=space, target=flat_space,
                         convert_from=_Lookup(lookup), convert_to=_Lookup(inverse_lookup))

    elif isinstance(space, spaces.Tuple):
        # first ensure all subspaces are flat.
        flat_subs = [flatten(sub) for sub in space.spaces]
        lo = np.concatenate([f.target.low for f in flat_subs])
        hi = np.concatenate([f.target.high for f in flat_subs])
        return Transform(space, target=spaces.Box(low=lo, high=hi), convert_to=_FlattenTuple(flat_subs),
                         convert_from=_DecomposeTuple(flat_subs))

    raise NotImplementedError("Does not know how to flatten {}".format(type(space)))  # pragma: no cover


# rescale a continuous action space
def rescale(space, low, high):
    """ A space transform that changes a continuous
        space to a new one with the specified upper and lower
        bounds by linear transformations. If source and target
        range are infinite, only the offset is corrected. If
        one of the ranges is finite and the other infinite, an
        error is raised.
    :rtype: Transform
    :param gym.Space space: The original space. Needs to be
        continuous.
    :param low: Lower bound of the new space.
    :param high: Upper bound of the new space.
    """
    if is_discrete(space):
        raise TypeError("Cannot rescale discrete space {}".format(space))

    if not isinstance(space, spaces.Box):
        raise NotImplementedError()

    # shortcuts
    lo = space.low
    hi = space.high

    # ensure new low/high values are arrays
    if isinstance(low, numbers.Number):
        low = np.ones_like(space.low) * low
    if isinstance(high, numbers.Number):
        high = np.ones_like(space.high) * high

    offset = np.copy(low)
    rg = hi - lo
    rs = high - low
    if np.isnan(rs).any():
        raise ValueError("Invalid range %s to %s specified" % (low, high))

    # the following code is responsible for correctly setting the scale factor and offset
    # in cases where the limits of the ranges become infinite.
    scale_factor = np.zeros_like(lo)
    for i in range(lo.size):
        if np.isinf(rg[i]) and np.isinf(rs[i]):
            scale_factor[i] = 1.0
        else:
            scale_factor[i] = rg[i] / rs[i]

        if low[i] == -np.inf and lo[i] == -np.inf:
            lo[i] = 0.0
            if high[i] == np.inf and hi[i] == np.inf:
                offset[i] = 0.0
            else:
                offset[i] = high[i] - hi[i]

    if np.isinf(scale_factor).any() or (scale_factor == 0.0).any():
        raise ValueError("Cannot map finite to infinite range [%s to %s] to [%s to %s] " % (lo, hi, low, high))

    def convert(x):
        y = (x - offset) * scale_factor # y is in [0, rg]
        return y + lo

    def back(x):
        return (x - lo) / scale_factor + offset

    scaled_space = spaces.Box(low, high, dtype=space.dtype)
    return Transform(original=space, target=scaled_space, convert_from=convert, convert_to=back)
