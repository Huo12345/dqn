import random
import numpy as np

from dqn.Game import Game


class PickGame(Game):

    def __init__(self, size=4):
        self.size = size
        self.state = np.zeros(self.size, dtype=np.float32)
        self.possible_actions = list(range(self.size))
        self.over = False

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)
        self.state[random.choice(range(self.size))] = 1
        self.over = False

    def move(self, action):
        if self.state[action] == 0:
            self.state[action] = 1
            self.over = all([e == 1 for e in self.state])
            return 1 if self.over else 0
        self.over = True
        return -1

    def actions(self):
        return self.possible_actions

    def is_over(self):
        return self.over

    def get_state(self):
        return np.copy(self.state)
