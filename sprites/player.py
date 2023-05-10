import pygame
from pygame.locals import (
    K_LEFT,
    K_RIGHT,
)

import constants


class Player(pygame.sprite.Sprite):
    rect = 0
    def __init__(self, sprites):
        super(Player, self).__init__()
        self.timer = 0
        self.interval = 2
        self.number_of_images = 6
        self.images = sprites.load_strip([0, 130, 48, 45], self.number_of_images, -1)
        self.surf = self.images[0]
        self.rect = self.surf.get_rect(center=(constants.SCREEN_WIDTH / 2, constants.SCREEN_HEIGHT - 40))
        self.image_index = 0

    def get_event(self, event):
        pass

    def update(self, pressed_keys):
        self.timer += 1

        if pressed_keys == 1 or pressed_keys == 3:
            self.rect.move_ip(-10, 0)
        if pressed_keys == 0 or pressed_keys == 2:
            self.rect.move_ip(10, 0)

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > constants.SCREEN_WIDTH:
            self.rect.right = constants.SCREEN_WIDTH

    @staticmethod
    def updateAI(pressed_keys):

        if pressed_keys == 1:
            Player.rect.move_ip(-5, 0)
        if pressed_keys == 2:
            Player.rect.move_ip(5, 0)

        if Player.rect.left < 0:
            Player.rect.left = 0
        if Player.rect.right > constants.SCREEN_WIDTH:
            Player.rect.right = constants.SCREEN_WIDTH

    def get_surf(self):
        if self.timer % self.interval == 0:
            self.image_index += 1
            if self.image_index >= self.number_of_images:
                self.image_index = 0
        return self.images[self.image_index]
