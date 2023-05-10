import numpy as np
import pygame
import torch
from torch.distributions import Categorical
from torch import nn, optim

import constants
from game import Game
from sprites.player import Player
from states.game_over import GameOver
from states.gameplay import Gameplay
from states.menu import Menu
from states.splash import Splash


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(512 * 768, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 actions: "r", "l", "rs", "ls"

    def forward(self, x):
        state = x.reshape(1, -1)
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)

        return action_probs

    def act(self, inputVal):
        print(inputVal)
        state = np.array(inputVal)
        state = torch.from_numpy(state).float().unsqueeze(0)
        print(state)
        action_probs = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()

        return action.item(), m.log_prob(action)


def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# setup mixer to avoid sound lag
pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((constants.SCREEN_WIDTH,
                                  constants.SCREEN_HEIGHT))
states = {
    "MENU": Menu(),
    "SPLASH": Splash(),
    "GAMEPLAY": Gameplay(),
    "GAME_OVER": GameOver(),
}

game = Game(screen, states, "SPLASH")
game.run()

# Initialize policy and optimizer
policy = PPO()
optimizer = torch.optim.Adam(policy.parameters())

# Hyperparameters
num_episodes = 10000000
clip_epsilon = 0.2
gamma = 0.99

# Training loop
for episode in range(num_episodes):
    log_probs = []
    rewards = []
    actions = []
    states = []

    # Initialize a new game

    import time

    start_time = time.time()
    while not Game.done:
        state = game.run()
        reward = time.time() - start_time

        action, log_prob = policy.act(state)

        log_probs.append(log_prob)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        print(action)
        constants.BTN = action

    returns = compute_returns(rewards, gamma)

    # Policy gradient update
    for state, action, old_log_prob, return_ in zip(states, actions, log_probs, returns):
        action, log_prob = policy.act(state)
        ratio = (log_prob - old_log_prob).exp()

        # Policy loss
        policy_loss = -torch.min(ratio * return_, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * return_)

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    # Log progress
    if episode % 10 == 0:
        print(f'Episode {episode}, Total reward: {sum(rewards)}')

# Save the trained model
torch.save(policy.state_dict(), 'path_to_your_trained_model.pt')
pygame.quit()
