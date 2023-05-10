import numpy as np
import pygame
import torch

import constants


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def buildFrame(screen):

    pass


class Game(object):
    binary_array = []
    done = False
    def __init__(self, screen, states, start_state):

        self.screen = screen
        self.clock = pygame.time.Clock()
        self.fps = constants.FPS
        self.states = states
        self.state_name = start_state
        self.state = self.states[self.state_name]
        self.binary_array = np.where(rgb2gray(pygame.surfarray.array3d(self.screen)[768 // 2:, :]) > 128, 1, 0).astype(np.float32)

    def event_loop(self):
        for event in pygame.event.get():
            self.state.get_event(event)

    def flip_state(self):
        next_state = self.state.next_state
        self.state.done = False
        self.state_name = next_state
        self.state = self.states[self.state_name]
        self.state.startup()

    def update(self, dt):
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        self.state.update(dt)

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.state.draw(self.screen)

    def run(self):
            dt = self.clock.tick(self.fps)

            state = pygame.surfarray.array3d(self.screen)[1024 // 2:, :]
            state = rgb2gray(state)
            state = np.where(state > 128, 1, 0)
            state = state.astype(np.float32)






            # Prepare the state for the network
            # Get the action from the network



            # Perform the chosen action in your game
            # This will depend on how your game is structured


            buildFrame(self.screen)
            self.event_loop()
            self.update(dt)
            self.draw()
            pygame.display.update()
            return state
