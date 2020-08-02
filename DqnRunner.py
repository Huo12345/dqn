import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datau

from CircularBufferDataset import CircularBufferDataset
from hello_world.FeedForwardNetwork import FeedForwardNetwork
from hello_world.Game import Game


def run_dqn():
    dataset = load_dataset()
    online_model, offline_model, criterion, optimizer, device = load_model()
    train_model(online_model, offline_model, criterion, optimizer, dataset, Game(), device)


def load_dataset():
    return CircularBufferDataset(500)


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    online_model = FeedForwardNetwork()
    offline_model = FeedForwardNetwork()
    offline_model.load_state_dict(online_model.state_dict())
    online_model.to(device)
    offline_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(online_model.parameters(), lr=0.001, momentum=0.9)
    print(device)
    return online_model, offline_model, criterion, optimizer, device


def train_model(online_model, offline_model, criterion, optimizer, trainset, game, device, batch_size=200, epochs=10, exploration_decay=0.9, discount_factor=0.9, test_rounds=20):
    exploration_factor = 0.9
    for epoch in range(epochs):
        print("Epoch %d/%d" % (epoch, epochs))

        generate_memories(offline_model, game, trainset, exploration_factor, batch_size, device)
        exploration_factor *= exploration_decay

        episode_set = datau.Subset(trainset, random.sample(range(trainset.__len__()), batch_size))
        memory_training(online_model, offline_model, criterion, optimizer, episode_set, device, discount_factor)
        offline_model.load_state_dict(online_model.state_dict())

        test_model(offline_model, game, test_rounds, device)

    print('Finished Training')


def generate_memories(model, game, buffer, exploration, size, device):
    with torch.no_grad():
        game.reset()
        for i in range(size):
            state = torch.tensor(game.get_state()).to(device)
            prediction = model(state).cpu()
            action = torch.argmax(prediction) if random.uniform(0, 1) > exploration else torch.tensor(random.choice(game.action()))
            reward = torch.tensor(game.move(action))
            next_state = torch.tensor(game.get_state())
            buffer.add_items([(state, action, reward, next_state)])
            if game.is_over():
                game.reset()


def memory_training(online_model, offline_model, criterion, optimizer, data_set, device, discount_factor):
    loader = datau.DataLoader(data_set, batch_size=4, shuffle=True)
    for data in loader:
        inputs, action, reward, next_state = [entry.to(device) for entry in data]
        optimizer.zero_grad()

        output = torch.gather(online_model(inputs), index=action.view(-1, 1), dim=1).view(-1)
        with torch.no_grad():
            expected_reward = offline_model(next_state) * discount_factor
        loss = criterion(output, reward + torch.max(expected_reward, dim=1)[0])
        loss.backward()
        optimizer.step()


def test_model(model, game, rounds, device):
    points = 0
    game.reset()
    with torch.no_grad():
        for r in range(rounds):
            action = torch.argmax(model(torch.tensor(game.get_state()).to(device))).cpu()
            points += game.move(action)
            if game.is_over():
                game.reset()
    print("Average points per round %.3f" % (float(points) / rounds))


if __name__ == '__main__':
    run_dqn()
