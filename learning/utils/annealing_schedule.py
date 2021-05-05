from abc import ABC, abstractmethod


class AnnealingSchedule(ABC):
    def __init__(self, starting_par: float, ending_par: float, n_steps: int, inverse=False):
        self.starting_p = starting_par
        self.ending_p = ending_par
        self.n_steps = n_steps
        self.current_p = starting_par

        self.inverse = inverse

    @abstractmethod
    def anneal(self):
        pass


class LinearSchedule(AnnealingSchedule):
    # linear annealing of a parameter
    def __init__(self, starting_par: float, ending_par: float, n_steps: int, inverse=False):
        super().__init__(starting_par, ending_par, n_steps, inverse=inverse)
        self.p_drop = (self.starting_p - self.ending_p) / self.n_steps

    def anneal(self):
        if not self.inverse:
            if self.current_p > self.ending_p:
                self.current_p = max(self.ending_p, self.current_p - self.p_drop)
        else:

            if self.current_p < self.ending_p:
                self.current_p = self.current_p - self.p_drop

        return self.current_p


class ExponentialSchedule(AnnealingSchedule):
    def __init__(self, starting_par: float, ending_par: float, n_steps: int, decay_rate: float, inverse=False):
        super().__init__(starting_par, ending_par, n_steps, inverse=inverse)
        self.decay_rate = decay_rate

    def decayed_learning_rate(self, step):
        return self.starting_p * self.decay_rate ** (step / self.n_steps)

if __name__ == '__main__':

    lin_schedule = LinearSchedule(0.1, 0.001, 1000)
    print(lin_schedule.p_drop)
