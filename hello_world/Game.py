import numpy as np


class Game:

    def __init__(self):
        self.size = 4
        self.state = np.zeros(self.size, dtype=np.float32)
        self.actions = list(range(self.size))

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def move(self, action):
        reward = 1 if self.state[action] == 0 else -1
        self.state[action] = 1
        return reward

    def action(self):
        return self.actions

    def is_over(self):
        return all([e == 1 for e in self.state])

    def get_state(self):
        return np.copy(self.state)
