import abc


class Game(abc.ABC):

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def move(self, action):
        pass

    @abc.abstractmethod
    def actions(self):
        pass

    @abc.abstractmethod
    def is_over(self):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass
