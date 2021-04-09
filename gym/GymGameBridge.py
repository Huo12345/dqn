import numpy as np
from gym.spaces import Discrete

from dqn.Game import Game


class GymGameBridge(Game):

    def __init__(self, env, state_parser=None, action_parser=None):
        self.env = env
        self.current_state = None
        self.done = False
        self.state_parser = state_parser if state_parser is not None else lambda x: x
        self.action_parser = action_parser if action_parser is not None else lambda x: x

    def reset(self):
        self.done = False
        self.current_state = self.env.reset()

    def move(self, action):
        a, reward, b, c = self.env.step(action.item())
        self.current_state = a
        self.done = b
        return reward

    def actions(self):
        return self.action_parser(self.env.action_space)

    def is_over(self):
        return self.done

    def get_state(self):
        return np.copy(self.state_parser(self.current_state))

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.close()


def DiscreteSpaceParser(space: Discrete):
    return list(range(space.n))
