import numpy as np

from dqn.Game import Game


class PickGame(Game):

    def __init__(self, size=4):
        self.size = size
        self.state = np.zeros(self.size, dtype=np.float32)
        self.possible_actions = list(range(self.size))

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def move(self, action):
        reward = 1 if self.state[action] == 0 else -1
        self.state[action] = 1
        return reward

    def actions(self):
        return self.possible_actions

    def is_over(self):
        return all([e == 1 for e in self.state])

    def get_state(self):
        return np.copy(self.state)
