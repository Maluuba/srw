"""
Environments
"""
import sys
import os

import click
import cv2
import numpy as np
from ale_python_interface import ALEInterface


class AtariEnv(object):
    def __init__(self, frame_skip=None, repeat_action_probability=0.0, state_shape=[84, 84], rom_path=None,
                 game_name='pong', random_state=None, rendering=False, record_dir=None, obs_showing=False,
                 channel_weights=[0.5870, 0.2989, 0.1140]):
        self.ale = ALEInterface()
        self.frame_skip = frame_skip
        self.state_shape = state_shape
        if random_state is None:
            random_state = np.random.RandomState(1234)
        self.rng = random_state
        self.channel_weights = channel_weights
        self.ale.setInt(b'random_seed', self.rng.randint(1000))
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.setBool(b'color_averaging', False)
        if rendering:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                self.ale.setBool(b'sound', False)  # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.ale.setBool(b'sound', True)
            self.ale.setBool(b'display_screen', True)
        if rendering and record_dir is not None:  # should be before loadROM
            self.ale.setString(b'record_screen_dir', record_dir.encode())
            self.ale.setString(b'record_sound_filename', os.path.join(record_dir, '/sound.wav').encode())
            self.ale.setInt(b'fragsize', 64)  # to ensure proper sound sync (see ALE doc)
        self.ale.loadROM(str.encode(rom_path + game_name + '.bin'))
        self.legal_actions = self.ale.getMinimalActionSet()
        self.nb_actions = len(self.legal_actions)
        (self.screen_width, self.screen_height) = self.ale.getScreenDims()
        self._buffer = np.empty((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        self.obs_showing = obs_showing

    def reset(self):
        self.ale.reset_game()
        return self.get_state()

    def step(self, action):
        reward = 0.0
        if self.frame_skip is None:
            num_steps = 1
        elif isinstance(self.frame_skip, int):
            num_steps = self.frame_skip
        else:
            num_steps = self.rng.randint(self.frame_skip[0], self.frame_skip[1])
        for i in range(num_steps):
            reward += self.ale.act(self.legal_actions[action])
        return self.get_state(), reward, self.ale.game_over(), {}

    def _get_image(self):
        self.ale.getScreenRGB(self._buffer)
        gray = self.channel_weights[0] * self._buffer[:, :, 0] + self.channel_weights[1] * self._buffer[:, :, 1] + \
           self.channel_weights[2] * self._buffer[:, :, 2]
        x = cv2.resize(gray, tuple(self.state_shape), interpolation=cv2.INTER_LINEAR)
        return x

    def get_state(self):
        return self._get_image()

    def get_lives(self):
        return self.ale.lives()


@click.command()
@click.option('--human/--no-human', default=False, help='Activates the flat agent.')
@click.option('--game', default='pong', help='Game to play.')
@click.option('--show/--no-show', default=False, help='Shows the observations.')
def test(human, game, show):
    env = AtariEnv(rom_path='aleroms/', game_name=game, frame_skip=4, rendering=True, obs_showing=show)
    term = False
    while not term:
        if not human:
            a = np.random.randint(0, env.nb_actions)
        else:
            a = int(input('Action >> '))
        obs, r, term, _ = env.step(a)
        print('state>> {0} | action>> {1} | reward>> {2} | lives>> {3}'.format(obs, a, r, env.get_lives()))


if __name__ == '__main__':
    test()
