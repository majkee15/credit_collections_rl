import numpy as np
from gym.spaces import Discrete
from dcc import AAV, Parameters


class TrainConfig():
    alpha = 0.9
    alpha_decay = 0.9997
    epsilon = 1.0
    epsilon_final = 0.05
    n_episodes = 20000
    warmup_episodes = 18000


class QlearningPolicy():
    def __init__(self, denv, denva, name, training=True, gamma=0.99, Q=None):
        """
        This Q-learning implementation only works on an environment with discrete
        action and observation space. We use a dict to memorize the Q-value.
        1. We start from state s and
        2.  At state s, with action a, we observe a reward r(s, a) and get into the
        next state s'. Update Q function:
            Q(s, a) += learning_rate * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        Repeat this process.
        """
        # super().__init__(env, name, gamma=gamma, training=training)
        self.denv = denv
        self.denva = denva
        self.gamma = gamma
        self.training = training
        assert isinstance(denva.action_space, Discrete)
        assert isinstance(denva.observation_space, Discrete)

        self.Q_table = Q
        self.actions = range(self.denva.action_space.n)
        self.n_states = denva.observation_space.n
        self.n_actions = denva.action_space.n
        self.lambd = 0.9
        self.state_visits = np.zeros(self.n_states)

    def build(self, optimal=True):
        self.Q_table = np.random.rand(self.n_states, self.n_actions)
        if optimal:
            aav = AAV(parameters=self.denva.params)
            for i in range(self.Q_table.shape[0]):
                cell = self.denv._ind_to_cont(i)
                for j in range(self.Q_table.shape[1]):
                   self.Q_table[i, j] = -aav.u(cell[0] + self.denva.action_val_bins[0][j], cell[1]) - \
                                        self.denva.action(j) * self.denva.params.c
        self.eligibility_table = np.zeros_like(self.Q_table)
        return self.Q_table

    def act(self, state, eps=0.1):
        """Pick best action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy.
        """
        if self.training and eps > 0. and np.random.rand() < eps:
            return self.denva.action_space.sample()

        # Pick the action with highest Q value.
        qvals = self.Q_table[state, :]
        max_q = max(qvals)

        # In case multiple actions have the same maximum Q value.
        actions_with_max_q = [a for a, q in enumerate(qvals) if q == max_q]
        return np.random.choice(actions_with_max_q)

    def _update_q_value(self, s, r, a, s_next, done, alpha):
        """
        Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        """
        max_q_next = np.max(self.Q_table[s_next, :])
        self.Q_table[s, a] += alpha * (r + self.gamma * max_q_next * (1.0 - done) - self.Q_table[s, a])

    def _update_Q_eli(self, s, r, a, s_next, done, alpha):
        max_q_next = np.max(self.Q_table[s_next, :])
        self.eligibility_table *= self.lambd
        self.eligibility_table[s, a] += 1
        self.Q_table += alpha * (r + self.gamma * max_q_next - self.Q_table[s, a]) * self.eligibility_table

    def train(self, config, verbose=False, eli=False):
        self.linalgnorm = []
        if eli:
            update_q = self._update_Q_eli
        else:
            update_q = self._update_q_value

        reward_history = []
        reward_averaged = []
        step = 0
        alpha = config.alpha
        eps = config.epsilon

        warmup_episodes = config.warmup_episodes or config.n_episodes
        eps_drop = (config.epsilon - config.epsilon_final) / warmup_episodes

        for n_episode in range(config.n_episodes):
            self.linalgnorm.append(np.linalg.norm(self.Q_table))
            ob = self.denva.reset()
            self.state_visits[ob] += 1
            done = False
            reward = 0.

            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, info = self.denva.step(a)
                self.state_visits[ob] += 1
                update_q(new_ob, r, a, new_ob, done, alpha)

                step += 1
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward_averaged.append(np.average(reward_history[-50:]))

            alpha *= config.alpha_decay
            if eps > config.epsilon_final:
                eps = max(config.epsilon_final, eps - eps_drop)

        #     if config.log_every_episode is not None and n_episode % config.log_every_episode == 0:
        #         # Report the performance every 100 steps
        #         print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} Qsize:{}".format(
        #             n_episode, step, np.max(reward_history),
        #             np.mean(reward_history[-10:]), alpha, eps, len(self.Q)))
        #
            if len(reward_history) % 100 == 0 and verbose:
                print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
                    len(reward_history), np.max(reward_history[-50:]), np.mean(reward_history[-50:])))
        #
        # data_dict = {'reward': reward_history, 'reward_avg50': reward_averaged}
        return reward_averaged

if __name__ == '__main__':
    p = Parameters()
    opt = AAV(p).u(p.lambda0,100)
    from learning.collections_env.discwrapper import DiscretizedActionWrapper, DiscretizedObservationWrapper
    from learning.collections_env.gymcollectionsenv import CollectionsEnv
    import matplotlib.pyplot as plt

    env = CollectionsEnv()
    denv = DiscretizedObservationWrapper(env)
    denva = DiscretizedActionWrapper(denv, 5)
    print(f'N states: {denva.observation_space.n}')
    config = TrainConfig()
    qlearning = QlearningPolicy(denv, denva, 'qlearn')
    qlearning.build(optimal=True)

    rew = qlearning.train(config, eli=False)
    plt.plot(rew)
    plt.axhline(-opt, color='red', linestyle='--')
    plt.show()

