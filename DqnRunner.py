import random

import torch
import torch.nn as nn
import torch.optim as optim

from dqn.CircularBufferDataset import CircularBufferDataset
from dqn.hello_world.FeedForwardNetwork import FeedForwardNetwork


def run_dqn():
    dataset = load_dataset()
    online_model, offline_model, criterion, optimizer = load_model()
    train_model(train, model, criterion, optimizer)
    test_model(test, model)


def load_dataset():
    return CircularBufferDataset(500)


def load_model():
    online_model = FeedForwardNetwork()
    offline_model = FeedForwardNetwork()
    offline_model.load_state_dict(online_model.state_dict())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(online_model.parameters(), lr=0.001, momentum=0.9)
    return online_model, offline_model, criterion, optimizer


def train_model(trainset, model, criterion, optimizer, batch_size=200, epochs=10, exploration_decay=0.9):
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainset, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished Training')


def generate_memories(model, game, buffer, exploration, size):
    with torch.no_grad():
        game.reset()
        for i in range(size):
            state = game.get_state()
            prediction = model(state)
            action = torch.max(prediction, 1) if random.uniform(0, 1) > exploration else random.choice(game.action())
            reward = game.move(action)
            next_state = game.get_state()
            buffer.add_items([(state, action, reward, next_state)])
            if game.is_over():
                game.reset()


def test_model(testset, model):
    pass


if __name__ == '__main__':
    run_dqn()
