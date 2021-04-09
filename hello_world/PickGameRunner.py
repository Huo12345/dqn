from dqn.Dqn import Dqn
from dqn.hello_world.FeedForwardNetwork import FeedForwardNetwork
from dqn.hello_world.PickGame import PickGame


def run_pickgame_trainig():
    online_model = FeedForwardNetwork()
    offline_model = FeedForwardNetwork()
    offline_model.load_state_dict(online_model.state_dict())
    dqn = Dqn(online_model, offline_model, PickGame(), exploration_decay=0.90)
    dqn.train_model(epochs=50)


if __name__ == '__main__':
    run_pickgame_trainig()
