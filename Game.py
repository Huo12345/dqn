import abc


class Game(abc.ABC):

    @abc.abstractmethod
    def initial_state(self):
        pass

    @abc.abstractmethod
    def all_actions(self):
        pass

    @abc.abstractmethod
    def actions(self, state):
        pass

    @abc.abstractmethod
    def move(self, state, action):
        pass

    @abc.abstractmethod
    def is_over(self, state):
        pass
