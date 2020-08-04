import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datau

from CircularBufferDataset import CircularBufferDataset
from hello_world.FeedForwardNetwork import FeedForwardNetwork
from hello_world.PickGame import PickGame


def run_dqn():
    dataset = load_dataset()
    online_model, offline_model, criterion, optimizer, device = load_model()
    train_model(online_model, offline_model, criterion, optimizer, dataset, PickGame(), device)


def load_dataset():
    return CircularBufferDataset(5000)


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


def train_model(online_model, offline_model, criterion, optimizer, trainset, game, device, batch_size=1000, epochs=30, exploration_decay=0.9, discount_factor=0.9, test_rounds=20):
    exploration_factor = 0.9
    for epoch in range(epochs):
        print("Epoch %d/%d" % (epoch + 1, epochs))

        generate_memories(offline_model, game, trainset, exploration_factor, batch_size, device)
        exploration_factor *= exploration_decay

        episode_set = datau.Subset(trainset, random.sample(range(trainset.__len__()), batch_size))
        memory_training(online_model, offline_model, criterion, optimizer, episode_set, device, discount_factor)
        offline_model.load_state_dict(online_model.state_dict())

        test_model(offline_model, game, test_rounds, device)

    print('Finished Training')


def generate_memories(model, game, buffer, exploration, size, device):
    with torch.no_grad():
        state = game.initial_state()
        for i in range(size):
            model_state = torch.tensor(state).to(device)
            prediction = model(model_state).cpu()
            action = torch.argmax(prediction) if random.uniform(0, 1) > exploration else torch.tensor(random.choice(game.actions(state)))
            reward, next_state = game.move(state, action)
            buffer.add_items([(model_state.cpu(), action, torch.tensor(reward), torch.tensor(next_state))])
            state = game.initial_state() if game.is_over(state) else next_state


def memory_training(online_model, offline_model, criterion, optimizer, data_set, device, discount_factor):
    loader = datau.DataLoader(data_set, batch_size=4, shuffle=True)
    for data in loader:
        inputs, action, reward, next_state = [entry.to(device) for entry in data]
        optimizer.zero_grad()

        output = torch.gather(online_model(inputs), index=action.view(-1, 1), dim=1).view(-1)
        with torch.no_grad():
            offline_prediction = offline_model(next_state)
            expected_reward = reward + torch.max(offline_prediction, dim=1)[0] * discount_factor
        loss = criterion(output, expected_reward)
        loss.backward()
        optimizer.step()


def test_model(model, game, rounds, device):
    points = 0
    state = game.initial_state()
    with torch.no_grad():
        for r in range(rounds):
            q_values = model(torch.tensor(state).to(device))
            action = torch.argmax(q_values).cpu()
            reward, state = game.move(state, action)
            points += reward
            if game.is_over(state):
                state = game.initial_state()
    print("Average points per round %.3f" % (float(points) / rounds))


if __name__ == '__main__':
    run_dqn()
