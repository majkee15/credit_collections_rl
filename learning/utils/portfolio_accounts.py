import numpy as np
import os

SEED = 1
np.random.seed(SEED)

n_acc = 50
llb = 0.2
lub = 3.
wlb = 50.
wub = 200.


def generate_portfolio(n_acc, seed=SEED):
    portfolio = np.array([[np.random.uniform(llb, lub), np.random.uniform(wlb, wub)] for _ in range(n_acc)])
    return portfolio


def load_acc_portfolio():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'sample_portfolio.npy')
    print(filename)
    return np.load(filename)


if __name__ == '__main__':
    portfolio = generate_portfolio(n_acc=n_acc)
    np.save('sample_portfolio.npy', portfolio)