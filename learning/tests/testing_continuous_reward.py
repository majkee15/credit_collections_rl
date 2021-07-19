from learning.collections_env.collections_env import CollectionsEnv

import numpy as np
import matplotlib.pyplot as plt

def generate_path(env):
    done = False
    state = env.reset()
    rew_path = []
    discrete_rew = []
    ls = []
    ws = []

    ls.append(state[0])
    ws.append(state[1])
    while not done:
        state, reward, done, _ = env.step(0.0)
        rew_path.append(reward)

       #  discrete_rew.append(reward_d)
        # if state[1] != ws[-1]:
        #     # jump has occured
        #     # discrete_rew.append((ws[-1] - state[1]) * np.exp(-env.params.rho * env.current_time))
        # else:
        #     discrete_rew.append(0)

        ls.append(state[0])
        ws.append(state[1])
    return ls, ws, rew_path



SEED = np.random.randint(0, 10e6)
env = CollectionsEnv(reward_shaping='continuous', starting_state=np.array([1.0, 200.]))
env_d = CollectionsEnv(reward_shaping='discrete', starting_state=np.array([1.0, 200.]))

# done = False
# state = env.reset()
# rew_path = []
# discrete_rew = []
# ls = []
# ws = []
#
# ls.append(state[0])
# ws.append(state[1])
# while not done:
#     state, reward, done, _ = env.step(0.0)
#     state_d, reward_d, done_d, _ = env_d.step(0.0)
#     rew_path.append(reward)
#
#     discrete_rew.append(reward_d)
#     # if state[1] != ws[-1]:
#     #     # jump has occured
#     #     # discrete_rew.append((ws[-1] - state[1]) * np.exp(-env.params.rho * env.current_time))
#     # else:
#     #     discrete_rew.append(0)
#
#     ls.append(state[0])
#     ws.append(state[1])



# print(discrete_rew)
if __name__ == '__main__':
    from learning.utils.latex_plot import save_fig

    plt.style.use('ggplot')
    SEED = 4366781
    np.random.seed(seed=SEED)
    ls, ws, rew_path = generate_path(env)

    np.random.seed(seed=SEED)
    lsd, wsd, rew_pathd = generate_path(env_d)

    # fig, ax = plt.subplots()
    # ax.plot(rew_path)
    # ax.plot(rew_pathd)
    # ax.set_title('Continuous Reward')
    # ax.set_xlabel('Step')
    # fig.show()

    fig, ax = plt.subplots()
    ax.plot(np.cumsum(rew_path))
    ax.plot(np.cumsum(rew_pathd))
    # ax.plot(np.cumsum(discrete_rew))
    ax.set_title('Reward path')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')

    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    fig.show()
    save_fig(fig, 'continuous_reward.pdf')
    # fig.savefig('example_1.pdf', format='pdf', bbox_inches='tight')




    # fig, ax = plt.subplots()
    # ax.plot(ls)
    # ax.plot(lsd)
    # ax.set_title('Lambda')
    # ax.set_xlabel('Step')
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(ws)
    # ax.plot(wsd)
    # ax.set_title('Balance')
    # ax.set_xlabel('Step')
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(ws, ls)
    # ax.plot(wsd, lsd)
    # ax.set_title('Lambda')
    # ax.set_xlabel('Balance')
    # fig.show()

    print(SEED)

    print(f'Accumulated reward continuous: {np.sum(rew_path)}')

    # print(f'Accumulated reward sparse: {np.sum(discrete_rew)}')

    # 4366781 - SEED that works
    # import tikzplotlib
    # tikzplotlib.save("test.tex")