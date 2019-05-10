import os
from copy import deepcopy
import time
import pygame
import numpy as np
import logging
import click

logger = logging.getLogger(__name__)


class BridgeEnv(object):
    def __init__(self, max_steps=10000, bridge_len=100, max_swimming_len=4, gameover_at_deadend=False,
                 state_mode='tabular', rendering=False, image_saving=False, render_dir=None, rng=None):
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        self.max_steps = max_steps
        self.bridge_len = bridge_len
        self.gameover_at_deadend = gameover_at_deadend
        self.legal_actions = [0, 1, 2]
        self.nb_actions = len(self.legal_actions)
        self.actions_probs = {0: [.99, 0., .01], 1: [0, 1, 0], 2: [0, 0, 1]}  # forward, backward, fall
        self.state = np.zeros((2, self.bridge_len), dtype=np.int32)
        self.tabular_state_shape = (self.bridge_len, max_swimming_len)
        self.pos_x = 0
        self.swimming = False
        self.max_swimming_len = max_swimming_len
        self.init_flag = True  # indicates if swimming is done (to avoid confusion with being at init by action "1").
        self.swim_step = [0, 0]  # [0]==timer, [1]==required steps to reach init point
        self.rewards = {'positive': 100.0, 'negative': -1.0, 'step': 0.0}
        self.state_mode = state_mode  # how the returned state look like ('pixel' or '1hot' or 'multi-head')
        self.scr_w = self.bridge_len
        self.scr_h = 2
        self.rendering_scale = 70
        self._rendering = rendering
        if rendering:
            self._init_pygame()
        self.image_saving = image_saving
        self.render_dir_main = render_dir
        self.render_dir = None
        self.state_shape = None
        self.step_id = 0
        self.game_over = False
        self.reset()

    @property
    def rendering(self):
        return self._rendering

    @rendering.setter
    def rendering(self, flag):
        if flag is True:
            if self._rendering is False:
                self._init_pygame()
                self._rendering = True
        else:
            self.close()
            self._rendering = False

    def _init_pygame(self):
        pygame.init()
        size = [self.rendering_scale * self.scr_w, self.rendering_scale * self.scr_h]
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Bridge Effect")

    def _init_rendering_folder(self):
        if self.render_dir_main is None:
            self.render_dir_main = 'render'
        if not os.path.exists(os.path.join(os.getcwd(), self.render_dir_main)):
            os.mkdir(os.path.join(os.getcwd(), self.render_dir_main))
        i = 0
        while os.path.exists(os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))):
            i += 1
        self.render_dir = os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))
        os.mkdir(self.render_dir)

    def reset(self):
        if self.image_saving:
            self._init_rendering_folder()
        self.game_over = False
        self.step_id = 0
        self.pos_x = 0
        self.swim_step = [0, 0]
        self.swimming = False
        self.init_flag = True
        return self.get_state()

    def close(self):
        if self.rendering:
            pygame.quit()

    def _move(self, action):
        assert action in self.legal_actions, 'Illegal action.'
        if action > 1:  # all actions >1 are the same
            action = 2
        reward = self.rewards['step']
        self.init_flag = False
        if not self.swimming:
            selector = self.rng.multinomial(1, self.actions_probs[action])  # from behaviour-prob of the taken action
            a = int(np.where(selector == 1)[0])
            if a == 0:  # forward
                self.pos_x += 1
                self.state[0, self.pos_x] = 1
                self.state[0, self.pos_x - 1] = 0
                if self.pos_x == self.bridge_len - 1:
                    reward = self.rewards['positive']
                    self.game_over = True
            elif a == 1:  # backward
                if self.pos_x != 0:
                    self.pos_x -= 1
                    self.state[0, self.pos_x] = 1
                    self.state[0, self.pos_x + 1] = 0
            else:  # fall
                self.swimming = True
                self.swim_step = [1, self.rng.randint(2, self.max_swimming_len)]
        else:
            if self.swim_step[0] == self.swim_step[1]:
                self.swimming = False
                self.state[0:, self.pos_x] = 0
                self.pos_x = 0
                self.swim_step = [0, 0]
                reward = self.rewards['negative']
                self.init_flag = True
                if self.gameover_at_deadend == True:
                    self.game_over = True
            else:
                self.swim_step[0] += 1
                self.state[1, self.pos_x] += 1
        return reward

    def step(self, action):
        if self.game_over:
            raise ValueError('Environment has already been terminated.')
        if self.step_id >= self.max_steps - 1:
            self.game_over = True
            return self.get_state(), 0., self.game_over, {}
        self.step_id += 1
        reward = self._move(action)
        return self.get_state(), reward, self.game_over, {}

    def get_state(self):
        if self.state_mode == 'pixel':
            return self.get_state_pixel()
        elif self.state_mode == 'tabular':
            return self.get_state_tabular()
        else:
            raise ValueError('State-mode is not known.')

    def get_state_tabular(self):
        return deepcopy((self.pos_x, self.swim_step[0]))

    def get_state_pixel(self):
        raise NotImplementedError

    def render(self, pause=0.2):
        if not self.rendering:
            return
        pygame.event.pump()
        self.screen.fill((0, 0, 0))
        size = [self.rendering_scale, self.rendering_scale]
        if self.init_flag:
            agent = pygame.Rect(0, 0, size[0], size[1])
            pygame.draw.rect(self.screen, (255, 255, 0), agent)
        elif not self.swimming:
            agent = pygame.Rect(self.rendering_scale * self.pos_x, 0, size[0], size[1])
            pygame.draw.rect(self.screen, (255, 255, 255), agent)
        else:
            agent = pygame.Rect(self.rendering_scale * self.pos_x, self.rendering_scale, size[0], size[1])
            color = (50 * self.swim_step[0], 50 * self.swim_step[0], 255)
            pygame.draw.rect(self.screen, color, agent)
        pygame.display.flip()
        if self.image_saving:
            self.save_image()
        time.sleep(pause)

    def save_image(self):
        if self.rendering and self.render_dir is not None:
            pygame.image.save(self.screen, self.render_dir + '/render' + str(self.step_id) + '.jpg')
        else:
            raise ValueError('env.rendering is False and/or environment has not been reset.')


@click.command()
@click.option('--save/--no-save', default=False, help='Saving rendering screen.')
def test(save):
    rng = np.random.RandomState(123)
    env = BridgeEnv(max_steps=200, bridge_len=10, state_mode='tabular', max_swimming_len=5, rendering=True,
                    image_saving=False, render_dir=None, rng=rng)
    env.reset()
    env.render()
    while not env.game_over:
        print('=' * 20)
        # action = rng.randint(0, 10)
        # print('action: ', action)
        action = int(input('action >> '))
        obs, r, term, info = env.step(action)
        env.render()
        print('pos: ', env.pos_x)
        print('swimming: ', env.swim_step)
        print('reward: ', r)
        print(obs)


if __name__ == '__main__':
    test()
