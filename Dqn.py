import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datau

from dqn.CircularBufferDataset import CircularBufferDataset
from dqn.Game import Game


class Dqn:

    def __init__(self,
                 online_model: nn.Module,
                 offline_model: nn.Module,
                 game: Game,
                 discount_factor=0.9,
                 initial_exploration_factor=0.9,
                 exploration_decay=0.9,
                 memory_size=5000,
                 learning_rate=0.001,
                 momentum=0.9,
                 demo=0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.online_model = online_model.to(self.device)
        self.offline_model = offline_model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(online_model.parameters(), lr=learning_rate, momentum=momentum)

        self.memory = CircularBufferDataset(memory_size)

        self.game = game

        self.discount_factor = discount_factor
        self.exploration_factor = initial_exploration_factor
        self.exploration_decay = exploration_decay
        self.demo = demo

    def train_model(self, batch_size=1000, epochs=30, test_rounds=20):
        print("Running training on device %s" % self.device)
        for epoch in range(epochs):
            print("Epoch %d/%d" % (epoch + 1, epochs))

            self.generate_memories(batch_size)
            self.exploration_factor *= self.exploration_decay

            episode_set = datau.Subset(self.memory, random.sample(range(self.memory.__len__()), batch_size))
            self.memory_training(episode_set)
            self.offline_model.load_state_dict(self.online_model.state_dict())

            self.test_model(test_rounds)

            if self.demo != 0 and epoch % self.demo == 0:
                self.demo_progress()

        print('Finished Training')
        if self.demo != 0:
            self.demo_progress()

    def generate_memories(self, number_of_elements):
        with torch.no_grad():
            self.game.reset()
            state = self.game.get_state()
            for i in range(number_of_elements):
                model_state = torch.tensor(state).to(self.device)
                prediction = self.offline_model(model_state.view(1, -1)).cpu()
                action = torch.argmax(prediction) \
                    if random.uniform(0, 1) > self.exploration_factor \
                    else torch.tensor(random.choice(self.game.actions()))
                reward = self.game.move(action)
                next_state = self.game.get_state()
                self.memory.add_items([(model_state.cpu(), action, torch.tensor(reward), torch.tensor(next_state))])
                if self.game.is_over():
                    self.game.reset()
                state = self.game.get_state()

    def memory_training(self, data_set):
        loader = datau.DataLoader(data_set, batch_size=4, shuffle=True)
        for data in loader:
            inputs, action, reward, next_state = [entry.to(self.device) for entry in data]
            self.optimizer.zero_grad()

            estimations = self.online_model(inputs)
            output = torch.gather(estimations, index=action.view(-1, 1), dim=1).view(-1)
            with torch.no_grad():
                offline_prediction = self.offline_model(next_state)
                expected_reward = reward + torch.max(offline_prediction, dim=1)[0] * self.discount_factor
            loss = self.criterion(output, expected_reward)
            loss.backward()
            self.optimizer.step()

    def test_model(self, rounds):
        games = 1
        points = 0
        self.game.reset()
        state = self.game.get_state()
        with torch.no_grad():
            for r in range(rounds):
                q_values = self.offline_model(torch.tensor(state).view(1, -1).to(self.device))
                action = torch.argmax(q_values).cpu()
                reward = self.game.move(action)
                points += reward
                if self.game.is_over():
                    self.game.reset()
                    games += 1
                state = self.game.get_state()
        print("Average points per game %.3f" % (float(points) / games))

    def demo_progress(self):
        self.game.reset()
        state = self.game.get_state()
        self.game.render()
        with torch.no_grad():
            while not self.game.is_over():
                q_values = self.offline_model(torch.tensor(state).view(1, -1).to(self.device))
                action = torch.argmax(q_values).cpu()
                self.game.move(action)
                state = self.game.get_state()
                self.game.render()
            self.game.terminate()
