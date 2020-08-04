import numpy as np

from dqn.Game import Game


class PickGame(Game):

    def __init__(self):
        self.size = 4

    def initial_state(self):
        return np.zeros(self.size, dtype=np.float32)

    def all_actions(self):
        return list(range(self.size))

    def actions(self, state):
        return self.all_actions()

    def move(self, state, action):
        reward = 1 if state[action] == 0 else -1
        state[action] = 1
        return reward, state

    def is_over(self, state):
        return all([e == 1 for e in state])

