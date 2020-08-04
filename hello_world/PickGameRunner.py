from Dqn import Dqn
from hello_world.FeedForwardNetwork import FeedForwardNetwork
from hello_world.PickGame import PickGame

if __name__ == '__main__':
    online_model = FeedForwardNetwork()
    offline_model = FeedForwardNetwork()
    offline_model.load_state_dict(online_model.state_dict())
    dqn = Dqn(online_model, offline_model, PickGame())
    dqn.train_model()
