# simulate an exponential controlled HP with control policy given by f(w) = lambda
import numpy as np
import matplotlib.pyplot as plt
from base import Base


class CHP(Base):

    def __init__(self, starting_balance, starting_intensity, marginal_cost=1.0, collection_horizon=None,
                 lambda_infty=None, kappa=None, delta10=None, delta11=None, control_function=None,
                 value_precision_thershold=0.001, rho=0.05):
        super().__init__(__class__.__name__)
        self.starting_intensity = starting_intensity
        self.starting_balance = starting_balance
        self.marginal_cost = marginal_cost
        self.collection_horizon = collection_horizon
        self.lambda_infty = lambda_infty
        self.kappa = kappa
        self.delta10 = delta10
        self.delta11 = delta11
        self.rho = rho
        self.value_precision_threshold = value_precision_thershold
        self.flag_controlled = True

        if control_function is None:
            self.flag_controlled = False

            def control_function(w):
                return np.zeros_like(w)

        self.control_function = control_function
        self.repayment_draw = np.random.rand
        self.logger.info(f'Instantiated Controlled Hawkes Process @ ')
        self.reset_state()

    def reset_state(self):
        # reset simulation state
        self.flag_simulated = False
        self.arrivals = np.array([])  # history of the process (tau_i, r_i)
        self.interarrivals = np.array([])
        self.relativerepayments = np.array([])
        self.balances = np.array([self.starting_balance])
        self.control_levels = np.array([self.control_function(self.starting_balance)])
        self.continuous_effort_duration = np.array([])
        self.continuous_effort_intensity = np.array([])
        self.continuous_effort_start_sustain = np.array([])
        self.intensities_minus = np.array([])
        self.total_cost = 0
        if self.starting_intensity < self.control_levels[-1]:
            # intensities after a jump i.e. \lambda_i = \lambda(\tau_i)
            self.starting_jump = self.control_function(self.starting_balance) - self.starting_intensity
            self.intensities_plus = np.array([self.control_levels[-1]])
        else:
            self.starting_jump = 0
            self.intensities_plus = np.array([self.starting_intensity])

    def drift(self, s, lambda_start):
        # deterministic draft
        return self.lambda_infty + (lambda_start - self.lambda_infty) * np.exp(-self.kappa * s)

    def controlled_intensity(self, s, n):
        # intensity of the ihp generating n+1 th arrival
        lambda_start = self.intensities_plus[n]
        control_level = self.control_levels[n]
        drifted = self.drift(s, lambda_start)
        return np.maximum(drifted, control_level)

    def t_equation(self, lambda_start, w_start):
        # equation for duration to reach a holding region
        try:
            numerator = (lambda_start - self.lambda_infty)
            denominator = (self.control_function(w_start)) - self.lambda_infty

            if lambda_start < self.control_function(w_start):
                return 0
            elif denominator < 0:
                return np.NaN
            else:
                t = 1 / self.kappa * \
                    np.log(numerator / denominator)
                return t
        except:
            print('warning')
            print(f'Numerator {(lambda_start - self.lambda_infty)}')
            print(f'Denominator {(self.control_function(w_start)) - self.lambda_infty}')

    def next_arrival(self):
        # generates interarrival times
        flag_finished = False
        s = 0
        t_to_sustain = self.t_equation(self.intensities_plus[-1], self.balances[-1])
        n = len(self.interarrivals)
        while not flag_finished:
            lstar = self.controlled_intensity(s, n)
            w = -np.log(np.random.rand()) / lstar
            s = s + w
            d = np.random.rand()
            potential_repayment = self.repayment_draw()
            potential_balance = self.balances[-1] * (1 - potential_repayment)
            intensity_at_jump = self.controlled_intensity(s, n)
            if d * lstar <= intensity_at_jump:
                sustain_drift_time = s - t_to_sustain
                if sustain_drift_time > 0:
                    self.continuous_effort_duration = np.append(self.continuous_effort_duration, sustain_drift_time)
                    self.continuous_effort_intensity = np.append(self.continuous_effort_intensity, intensity_at_jump)
                    self.continuous_effort_start_sustain = np.append(self.continuous_effort_start_sustain,
                                                                     t_to_sustain + sum(self.interarrivals))

                self.control_levels = np.append(self.control_levels, self.control_function(potential_balance))
                self.intensities_minus = np.append(self.intensities_minus, intensity_at_jump)
                self.intensities_plus = np.append(self.intensities_plus, intensity_at_jump + self.delta10 +
                                                  self.delta11 * potential_repayment)
                self.interarrivals = np.append(self.interarrivals, s)
                self.relativerepayments = np.append(self.relativerepayments, potential_repayment)
                self.balances = np.append(self.balances, potential_balance)
                flag_finished = True
        # if n == 0 and self.starting_jump>0:
        #     t_to_sustain = 0
        #     sustain_drift_time = s
        # else:

    def simulate(self):
        self.next_arrival()
        # check if last arrival is not greater than a collection horizon
        # and the discounted value of the account is still important
        repayment_value_condition = True
        collection_horizon_condition = True
        while collection_horizon_condition & repayment_value_condition:
            self.next_arrival()
            dv_last_repayment = np.abs(np.diff(self.balances)[-1] * np.exp(-np.sum(self.interarrivals) * self.rho))
            repayment_value_condition = dv_last_repayment > self.value_precision_threshold
            collection_horizon_condition = np.sum(self.interarrivals) <= self.collection_horizon
        # self.logger.info(dv_last_repayment)
        self.arrivals = np.cumsum(self.interarrivals)
        self.arrivals = self.arrivals[:-1]
        self.interarrivals = self.interarrivals[:-1]
        self.relativerepayments = self.relativerepayments[:-1]
        self.intensities_plus = self.intensities_plus[:-1]
        self.balances = self.balances[:-1]
        self.control_levels = self.control_levels[:-1]
        self.flag_simulated = True
        self.calculate_costs()

    def calculate_costs(self, verbose=False):
        if verbose:
            self.logger.info(
                f'Jumps: {self.starting_jump} \n Continuous effort durations: {self.continuous_effort_duration}'
                f'\n Continuous effort starts: {self.continuous_effort_start_sustain}')
        total_costs = self.starting_jump + np.sum(np.multiply(self.continuous_effort_duration,
                                                              self.continuous_effort_intensity))
        self.total_cost = total_costs * self.marginal_cost
        pass

    def calculate_value_single(self, _=None, verbose=False):
        # discounted value of repayments
        self.reset_state()
        self.simulate()
        dc_revenue = np.sum(np.multiply(np.exp(- self.arrivals * self.rho), -np.diff(self.balances)))
        if verbose:
            self.logger.info(f'Revenues: {dc_revenue} \n Costs: {self.total_cost}')
        return dc_revenue + self.total_cost

    def calculate_value(self, mc_iteration=10000):
        running_sum = 0
        for i in range(mc_iteration):
            running_sum += self.calculate_value_single()
        return running_sum / mc_iteration

    # PLOT FUNCTIONS

    def plot_intensity(self, dt=0.05):
        tgrid = np.arange(0, self.collection_horizon, dt)
        lambdas = np.empty_like(tgrid)
        prev_arrival = 0
        fig, ax = plt.subplots(1, 1)
        segment_time_points = np.append(self.arrivals, self.collection_horizon)
        for i, arrival in enumerate(segment_time_points):  # enumerate(self.intensities_plus):
            mask = [(tgrid >= prev_arrival) & (tgrid <= arrival)]
            ts = tgrid[tuple(mask)] - prev_arrival
            lambdas[tuple(mask)] = self.controlled_intensity(ts, i)
            prev_arrival = segment_time_points[i]
            # ax.plot(tgrid, self.controlled_intensity(tgrid, i), linestyle='--', color='black', linewidth=0.7)
        ax.plot(tgrid, lambdas)
        ax.axhline(self.lambda_infty, color='red', linestyle='--')
        fig.show()

    def plot_drift(self, lambdastart, dt=0.05, tmax=20):
        tgrid = np.arange(0, tmax, dt)
        lambdagrid = self.drift(tgrid, lambdastart)
        fig, ax = plt.subplots(1, 1)
        ax.plot(tgrid, lambdagrid)
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        ax.set_ylim([0, self.lambda_infty * 4])
        ax.axhline(self.lambda_infty, linewidth=1.0, color='r', linestyle='--')
        fig.show()
        return fig

    def plot_statespace(self):
        color = 'C0'
        if self.flag_simulated:
            fig, ax = plt.subplots(1, 1)
            ax.axhline(self.lambda_infty, linewidth=1.0, color='r', linestyle='--')
            if self.control_function is not None:
                if self.starting_jump > 0:
                    ax.plot([self.starting_balance, self.starting_balance],
                            [self.starting_intensity, self.starting_intensity + self.starting_jump],
                            c=color, linewidth=1.5, marker='x', linestyle='--')
                wgrid = np.arange(0, self.starting_balance, 1)
                ax.plot(wgrid, self.control_function(wgrid), color='black', linewidth=0.5)
                # ax.plot(self.balances, self.intensities_plus, marker='x', color='r')
                # ax. plot(self.balances[:-1], self.intensities_minus, marker='x', color='b')
                # y = np.array(list(chain.from_iterable(zip(self.intensities_plus, self.intensities_minus))))
                # x = np.array(list(chain.from_iterable(zip(self.balances, self.balances[:-1]))))
            starts = np.column_stack([self.balances, self.intensities_plus])
            stops = np.column_stack([self.balances, self.intensities_minus])
            jump = None
            for start, stop in zip(starts, stops):
                if jump is not None:
                    ix, iy = zip(jump, start)
                    ax.plot(ix, iy, c=color, linewidth=1.5, marker='x', linestyle='--')
                x, y = zip(start, stop)
                ax.plot(x, y, c=color, linewidth=1.5, marker='x')
                jump = stop
            plt.show()
            return fig
        else:
            self.logger.info('Process not simulated.')

    def plot_policy(self):
        if self.flag_controlled:
            ylim = np.maximum(self.starting_jump, self.control_levels[0]) * 1.5
            xlim = self.collection_horizon + self.collection_horizon * 0.01
            fig, ax = plt.subplots()
            print(self.starting_jump)
            head_length = ylim * 0.05
            head_width = xlim * 0.01
            ax.arrow(0, 0, 0, self.starting_jump - head_length,
                     head_width=head_width, head_length=head_length, fc='k', ec='k')

            for i, sustain in enumerate(self.continuous_effort_start_sustain):
                ax.plot([sustain, sustain + self.continuous_effort_duration[i]],
                        [self.continuous_effort_intensity[i], self.continuous_effort_intensity[i]], marker='x')

            ax.set_xlim(-self.collection_horizon * 0.01, self.collection_horizon)
            ax.set_ylim(0, ylim)
            ax.set_ylabel('Intensity')
            ax.set_xlabel('Time')
            ax.set_title('Derivative Policy')
            fig.show()
            return fig

    def plot_balance(self, dt=0.5):
        if self.flag_simulated:
            fig, ax = plt.subplots()
            tgrid = np.insert(self.arrivals, 0, 0)
            ax.step(self.balances, tgrid)
            ax.set_ylabel('Time')
            ax.set_xlabel('Balance')
            ax.set_xlim([0, self.starting_balance * 1.1])
            fig.show()

    # TESTS

    # TODO: test for lambda_i > self.control - otherwise jump would be necessary
    def test_increments(self):
        theoretical = np.round(chp.intensities_plus[1:] - chp.intensities_minus[:-1], 6)
        realized = np.round(self.relativerepayments * self.delta11 + self.delta10, 6)
        if np.array_equal(theoretical, realized):
            self.logger.info('Passed.')
        else:
            self.logger.info(f'Failed to match increments.\n Realized: {realized} \n Theoretical: {theoretical}')


if __name__ == '__main__':

    def control(w):
        shift = 50
        if isinstance(w, np.ndarray):
            return (0.001 * (w - shift)).clip(min=0)
        elif np.isscalar(w):
            return np.maximum(0.001 * (w - shift), 0)
        else:
            raise ValueError('Stupid error in w.')

    #
    chp = CHP(starting_balance=1000, starting_intensity=0.3, marginal_cost=10,
              collection_horizon=100, lambda_infty=0.1, kappa=1, value_precision_thershold=0.0,
              delta10=0.05, delta11=0.5, control_function=control, rho=0.05)
    chp.calculate_value_single()
    chp.plot_statespace()
    chp.plot_intensity()
    chp.plot_policy()
    chp.plot_balance()
    chp.test_increments()

    # print(chp.calculate_value(10000))
    # w_grid = np.arange(0, 100, 10)
    # v = np.zeros_like(w_grid)
    # for i, w in enumerate(w_grid):
    #     chp = CHP(starting_balance=w, starting_intensity=1, marginal_cost=1,
    #               collection_horizon=100, lambda_infty=0.1, kappa=0.7, value_precision_thershold=0.001,
    #               delta10=0.02, delta11=0.5, control_function=None, rho=0.06)
    #     v[i] = chp.calculate_value(mc_iteration=1000)
    #
    # plt.plot(w_grid, -v, marker='o')
    # plt.xlim([0,max(w_grid)])
    # plt.xlabel('w')
    # plt.ylabel('v')
    # plt.show()
    #
    # lamdba_grid = np.arange(0.1, 5, 1)
    # v = np.zeros_like(lamdba_grid)
    # for i, lam in enumerate(lamdba_grid):
    #     chp = CHP(starting_balance=75, starting_intensity=lam, marginal_cost=1,
    #               collection_horizon=100, lambda_infty=0.1, kappa=0.7, value_precision_thershold=0.001,
    #               delta10=0.02, delta11=0.5, control_function=None, rho=0.06)
    #     v[i] = chp.calculate_value(mc_iteration=10000)
    # print(v)
    # plt.plot(lamdba_grid, -v, marker='o')
    # plt.xlim([0, max(lamdba_grid)])
    # plt.xlabel('lambda')
    # plt.ylabel('v')
    # plt.show()
