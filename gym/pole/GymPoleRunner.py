import gym

from dqn.Dqn import Dqn
from dqn.gym.GymGameBridge import GymGameBridge, DiscreteSpaceParser
from dqn.gym.pole.FeedForwardNetwork import FeedForwardNetwork


def run_pole():
    env = gym.make('CartPole-v0')
    online_model = FeedForwardNetwork()
    offline_model = FeedForwardNetwork()
    offline_model.load_state_dict(online_model.state_dict())
    game = GymGameBridge(env=env, action_parser=DiscreteSpaceParser)
    dqn = Dqn(online_model, offline_model, game)
    dqn.train_model()
    game.terminate()


if __name__ == '__main__':
    run_pole()
    # env = gym.make('CartPole-v0')
    # state = env.reset()
    # print(state)
