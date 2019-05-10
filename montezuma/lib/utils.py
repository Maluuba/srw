"""
Utilities
"""

import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    if all([len(data[key]) > 1 for key in data]):
        if moving_average:
            smoothed_data = {}
            for key in data:
                smooth_scores = [np.mean(data[key][max(0, i - 10):i + 1]) for i in range(len(data[key]))]
                smoothed_data['smoothed_' + key] = smooth_scores
                smoothed_data[key] = data[key]
            data = smoothed_data
        df = pd.DataFrame(data=data)
        ax = df.plot(kind=kind, legend=legend, ylim=clip)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(loc)
        plt.close()


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)


class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


class ExperienceReplay(object):
    def __init__(self, max_size=100, history_len=1, state_shape=None, action_dim=1, reward_dim=1, state_dtype=np.uint8):
        self.size = 0
        self.head = 0
        self.tail = 0
        self.max_size = max_size
        self.history_len = history_len
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.state_dtype = state_dtype
        self._minibatch_size = None
        self.states = np.zeros([self.max_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.max_size, dtype='bool')
        if self.action_dim == 1:
            self.actions = np.zeros(self.max_size, dtype='int32')
        else:
            self.actions = np.zeros((self.max_size, self.action_dim), dtype='int32')
        if self.reward_dim == 1:
            self.rewards = np.zeros(self.max_size, dtype='float32')
        else:
            self.rewards = np.zeros((self.max_size, self.reward_dim), dtype='float32')

    def _init_batch(self, number):
        self.s = np.zeros([number] + [self.history_len] + list(self.state_shape), dtype=self.states[0].dtype)
        self.s2 = np.zeros([number] + [self.history_len] + list(self.state_shape), dtype=self.states[0].dtype)
        self.t = np.zeros(number, dtype='bool')
        action_indicator = self.actions[0]
        if self.actions.ndim == 1:
            self.a = np.zeros(number, dtype='int32')
        else:
            self.a = np.zeros((number, action_indicator.size), dtype=action_indicator.dtype)
        if self.rewards.ndim == 1:
            self.r = np.zeros(number, dtype='float32')
        else:
            self.r = np.zeros((number, 2), dtype='float32')

    def sample(self, num=1):
        if self.size == 0:
            logging.error('cannot sample from empty transition table')
        elif num <= self.size:
            if not self._minibatch_size or num != self._minibatch_size:
                self._init_batch(number=num)
                self._minibatch_size = num
            for i in range(num):
                self.s[i], self.a[i], self.r[i], self.s2[i], self.t[i] = self._get_transition()
            return self.s, self.a, self.r, self.s2, self.t
        elif num > self.size:
            logging.error('transition table has only {0} elements; {1} requested'.format(self.size, num))

    def _get_transition(self):
        sample_success = False
        while not sample_success:
            randint = np.random.randint(self.head, self.head + self.size - self.history_len)
            state_indices = np.arange(randint, randint + self.history_len)
            next_state_indices = state_indices + 1
            transition_index = randint + self.history_len - 1
            a_axis = None if self.action_dim == 1 else 0
            r_axis = None if self.reward_dim == 1 else 0
            if not np.any(self.terms.take(state_indices[:-1], mode='wrap')):
                s = self.states.take(state_indices, mode='wrap', axis=0)
                a = self.actions.take(transition_index, mode='wrap', axis=a_axis)
                r = self.rewards.take(transition_index, mode='wrap', axis=r_axis)
                t = self.terms.take(transition_index, mode='wrap')
                s2 = self.states.take(next_state_indices, mode='wrap', axis=0)
                sample_success = True
        return s, a, r, s2, t

    def add(self, s, a, r, t):
        self.states[self.tail] = s
        self.actions[self.tail] = a
        self.rewards[self.tail] = r
        self.terms[self.tail] = t
        self.tail = (self.tail + 1) % self.max_size
        if self.size == self.max_size:
            self.head = (self.head + 1) % self.max_size
        else:
            self.size += 1

    def reset(self):
        self.size = 0
        self.head = 0
        self.tail = 0
        self._minibatch_size = None
        self.states = np.zeros([self.max_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.max_size, dtype='bool')
        if isinstance(self.action_dim, int):
            self.actions = np.zeros(self.max_size, dtype='int32')
        else:
            self.actions = np.zeros((self.max_size, self.action_dim.size), dtype=self.action_dim.dtype)
        if isinstance(self.reward_dim, int):
            self.rewards = np.zeros(self.max_size, dtype='float32')
        else:
            self.rewards = np.zeros((self.max_size, 2), dtype='float32')
