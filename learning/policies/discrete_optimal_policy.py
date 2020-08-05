import joblib
import numpy as np


class DiscretePolicyConstructor:

    def __init__(self, env, mc_n):
        self.env = env
        self.n_actions = env.action_space.n
        self.mc_n = mc_n
        n_pts_w = int(200)
        n_pts_l = int(50)
        MAX_L = 7.0
        self.w_grid = np.linspace(0, self.env.w0, n_pts_w)
        self.l_grid = np.linspace(self.env.params.lambdainf, MAX_L, n_pts_l)
        self.ww, self.ll = np.meshgrid(self.w_grid, self.l_grid)
        self.policy = np.zeros_like(self.ww, dtype=np.int32)

    def get_action(self, state):
        bal = state[1]
        lam = state[0]
        i = np.digitize(bal, self.w_grid, right=True)
        j = np.digitize(lam, self.l_grid, right=True)
        return self.policy[j, i]

    def evaluate(self, start_state, first_action):
        self.env.reset(tostate=start_state)
        done = False
        score = 0

        next_state, reward, done, info = self.env.step(first_action)
        score += reward
        state = next_state.copy()

        while not done:
            action = self.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            score += reward
            state = next_state.copy()
        return score

    def parallel_evaluate(self, state, first_ac, mc):
        res = joblib.Parallel(n_jobs=6)(joblib.delayed(self.evaluate)(state, first_ac) for i in range(mc))
        return np.mean(res)

    def run(self):
        action_estimates = np.zeros(self.n_actions)
        i = 0
        for j, w in enumerate(self.w_grid):
            print(f'Balance: {w}')
            flag_searching = True
            while flag_searching:
                l = self.l_grid[i]
                state = np.array([l, w])
                for a_index in range(self.n_actions):
                    action_estimates[a_index] = self.parallel_evaluate(state, a_index, self.mc_n)

                best_action = np.argmax(action_estimates)

                if best_action == 0:
                    flag_searching = False
                else:
                    self.policy[i, j:] = best_action
                    i += 1
                    print(i)