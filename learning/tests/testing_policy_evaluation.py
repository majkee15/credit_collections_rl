from learning.collections_env import CollectionsEnv
from learning.utils.wrappers import StateNormalization, DiscretizedActionWrapper

import numpy as np
from copy import deepcopy

action_bins = np.array([0, 1.0])

environment = CollectionsEnv(reward_shaping=False)
environment = DiscretizedActionWrapper(environment, action_bins)
# environment = StateNormalization(environment)

print(f"Reseting: {environment.reset()}")
print(f"Step: {environment.step(0)}")
print(f"Reseting: {environment.reset()}")

environment.env.starting_state = 2#np.array([10, 100])
print(f"Setting start: {environment.env.starting_state}")
print(f"Reseting: {environment.reset()}")

#
# class ImuT:
#     def __init__(self, a):
#         self.a = np.array([0])
#         self.starting_a = self.a.copy()
#
#     def reset(self):
#         self.a = self.starting_a.copy()
#         return self.a
#
#     def step(self):
#         self.a = self.a + 1
#
#     def __repr__(self):
#         return f'A is: {self.a}.'
#
#
# if __name__ == '__main__':
#     a = 0
#     imut = ImuT(a)
#     print(imut)
#     imut.a = np.array([2, 1])
#     print(f'Imut value: {imut.a}')
#     imut.step()
#     print(imut)
#     print(f'Imut reset: {imut.reset()}')
#     print(imut)
#
