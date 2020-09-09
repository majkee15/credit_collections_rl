import numpy as np
from abc import ABC, abstractmethod


class RepaymentDist(ABC):
    def __init__(self, params):
        self.params = params
        self._mean = self.mean

    @abstractmethod
    def draw(self, w):
        raise NotImplementedError

    @abstractmethod
    def mean(self, w):
        raise NotImplementedError


class UniformRepayment(RepaymentDist):
    def __init__(self, params):
        super().__init__(params)

    def draw(self, w):
        return self.params.sample_repayment()[0]

    def mean(self, w):
        return 0.5 * (self.params.r_ + 1)


class BetaRepayment(RepaymentDist):

    def __init__(self, params, rmin, rmax, wmin, wmax):
        """

        Args:
            params: Parameter class from dcc module
            rmin: expected repayment value for the wmin balance
            rmax:expected repayment value for the wmax balance
            wmin: balance below repayment is always full
            wmax: maximal allowed balance
        """
        super().__init__(params)
        self.rmin = rmin
        self.rmax = rmax
        self.wmin = wmin
        self.wmax = wmax

    def f(self, w):
        return np.divide(w - self.wmin, self.wmax - self.wmin)

    def g(self, x):
        return (1/self.rmax - 1/self.rmin) * x + 1/self.rmin - 1

    def beta(self, w):
        return self.g(self.f(w))

    def draw(self, w):
        if w <= self.wmin:
            return 1
        else:
            return np.random.beta(1, self.beta(w), 1)[0]

    def mean(self, w):
        return 1/(1 + self.beta(w))
