import os
import time
from copy import deepcopy
import numpy as np
import logging
from lib.utils import Font, write_to_csv, plot

logger = logging.getLogger(__name__)


class DQNExperiment(object):
    def __init__(self, env, ai, ai_explore, episode_max_len, q_threshold, history_len=1, max_start_nullops=1,
                 replay_min_size=0, score_window_size=100, folder_location='/experiments/', folder_name='expt',
                 epsilon=1.0, annealing=True, final_epsilon=0.1, annealing_start=1, annealing_steps=10000,
                 secure=False, max_secure=False, ai_rewarding_buffer_size=20, ai_explore_rewarding_buffer_size=100,
                 exploration_learning_steps=1000000, use_expl_inc=False, testing=False, rng=None):
        self.rng = rng
        self.q_threshold = q_threshold
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_computer = 0
        self.score_agent = 0
        self.eval_scores = []
        self.eval_steps = []
        self.env = env
        self.ai = ai
        self.ai_explore = ai_explore
        self.secure = secure
        self.max_secure = max_secure  # use security threshold when greedy and take random action if insecure greedy.
        self.ai_rewarding_buffer_size = ai_rewarding_buffer_size
        self.ai_explore_rewarding_buffer_size = ai_explore_rewarding_buffer_size
        self.exploration_learning_steps = exploration_learning_steps
        self.exploration_learning = True
        self.use_expl_inc = use_expl_inc
        self.history_len = history_len
        self.annealing = annealing
        self.epsilon = deepcopy(epsilon)
        self.start_epsilon = deepcopy(epsilon)
        self.final_epsilon = final_epsilon
        self.annealing_start = annealing_start
        self.annealing_steps = annealing_steps
        self.max_start_nullops = max_start_nullops
        if not testing:
            self.folder_name = self._create_folder(folder_location, folder_name)
            os.mkdir(self.folder_name + '/ai')
        self.episode_max_len = episode_max_len
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)
        self.goal_ = 0

    def do_epochs(self, number=1, steps_per_epoch=10000, is_learning=True, is_testing=True, steps_per_test=10000):
        for epoch in range(number):
            print('=' * 30)
            print(Font.green + Font.bold + '>>>>> Epoch  ' + str(epoch) + '/' + str(number - 1) + '  >>>>>' + Font.end)
            steps = 0
            while steps < steps_per_epoch:
                self.do_episodes(number=1, is_learning=is_learning)
                steps += self.last_episode_steps
            if is_testing:
                self.eval_scores.append(np.mean(self.score_agent_window))
                self.eval_steps.append(np.mean(self.steps_agent_window))
                self._plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                                     x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                     moving_average=True)
                self._plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
                                     x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
                if epoch % 10 == 0:
                    self.ai.dump_network(weights_file_path=self.folder_name + '/ai/q_network_weights_' + str(epoch) + '.h5',
                                         overwrite=True)
                    if self.secure and self.exploration_learning:
                        self.ai_explore.dump_network(weights_file_path=self.folder_name + '/ai/q_explore_network_weights_' +
                                                                   str(epoch) + '.h5', overwrite=True)
                # all_rewards.append(eval_scores / eval_episodes)
                if self.total_training_steps >= self.exploration_learning_steps:
                    self.exploration_learning = False
                    self.q_threshold = -0.15
        return []

    def do_episodes(self, number=1, is_learning=True):
        all_rewards = []
        for num in range(number):
            print('=' * 30)
            print(Font.darkcyan + Font.bold + '::Episode::  ' + Font.end + str(self.episode_num))
            reward = self._do_episode(is_learning=is_learning)
            all_rewards.append(reward)
            self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
            self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            print_string = ("\nSteps: {0} | Fps: {1} | Eps: {2} | Score: {3} | Moving Avg: {4} | "
                            "Moving Steps: {5} | Total Steps: {6} | Achieved Goals: {7}")
            print(print_string.format(self.last_episode_steps, self.fps, round(self.epsilon, 2),
                                      round(self.score_agent, 2), round(np.mean(self.score_agent_window), 2),
                                      np.mean(self.steps_agent_window), self.total_training_steps, self.goal_))
            self.episode_num += 1
        return all_rewards

    def evaluate(self, number=10, explore=True):
        for num in range(number):
            _ = self._do_episode(is_learning=False, evaluate=True, explore=explore)
            print_string = '\nSteps: {0} | Fps: {1} | Score: {2}'
            print(print_string.format(self.last_episode_steps, self.fps, round(self.score_agent, 2)))
        return self.score_agent

    def do_human_episode(self):
        rewards = []
        self.env.reset()
        self._reset()
        term = False
        while not term:
            self.last_episode_steps += 1
            prev_lives = self.env.get_lives()
            print('='*50)
            q = self.ai.get_q(self.last_state, target=False)[0]
            q_max = np.max(q)
            argmax_q = np.where(q == q_max)[0]
            q_text = ''
            for item in q:
                if item == q_max:
                    t = Font.bold + Font.yellow + str(round(item, 2)) + ' ' + Font.end
                    q_text += t
                else:
                    t = str(round(item, 2)) + ' '
                    q_text += t
            print(Font.bold + Font.yellow + 'Exploit Q >> ' + Font.end, q_text)
            print(Font.bold + Font.yellow + 'Greedy action >>  ' + str(argmax_q) + Font.end)
            if self.secure:
                print(Font.bold + Font.cyan + 'Explore Q >> ' + Font.end,
                      self.ai_explore.get_q(self.last_state, target=False)[0])
            action = input('action >> ')
            if action == '':
                continue
            if action == 'q':
                return 0
            action = int(action)
            if action >= self.env.nb_actions:
                logger.warning('Unknown action.')
                continue
            new_obs, reward, game_over, _ = self.env.step(action)
            print('reward: ', reward)
            if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
                new_obs = new_obs.reshape(self.env.state_shape)
            term = game_over  # or self.env.get_lives() < prev_lives
            self._update_state(new_obs)
            rewards.append(reward)
            if not term and self.last_episode_steps >= self.episode_max_len:
                logger.warning('Reaching maximum number of steps in the current episode.')
                term = True
        return rewards

    def _do_episode(self, is_learning=True, evaluate=False, explore=True):
        self.env.reset()
        rewards = []
        self._reset()
        term = False
        self.fps = 0
        start_time = time.time()
        while not term:
            reward, term = self._step(evaluate=evaluate, explore=explore)
            rewards.append(reward)
            # NOTE: train the main network only after at least one reward has been seen
            if self.ai.transitions.size >= self.replay_min_size and is_learning and \
               self.last_episode_steps % self.ai.learning_frequency == 0 and len(self.ai.rewarding_transitions) >= 1:
                self.ai.learn(goal_prob=0.1)
            if self.secure and self.exploration_learning:
                if self.ai_explore.transitions.size >= self.replay_min_size and is_learning and \
                   self.last_episode_steps % self.ai_explore.learning_frequency == 0:
                    self.ai_explore.learn(goal_prob=1.0)
            self.score_agent += reward
            if not term and self.last_episode_steps >= self.episode_max_len:
                logger.warning('Reaching maximum number of steps in the current episode.')
                term = True
                # increase exploration to prevent getting stuck (only after main annealing period)
                if self.use_expl_inc and self.annealing and self.total_training_steps >= self.annealing_steps:
                    self.epsilon = 0.8
                    self.start_epsilon = 0.8
                    self.annealing_start = deepcopy(self.total_training_steps)
                    self.annealing_steps = 10 * self.episode_max_len
        self.fps = int(self.last_episode_steps * self.env.frame_skip / max(time.time() - start_time, 0.01))
        return rewards

    def _step(self, evaluate=False, explore=True):
        self.last_episode_steps += 1
        prev_lives = self.env.get_lives()
        if bool(self.rng.binomial(1, self.epsilon)) and explore:  # safe exploration
            if self.secure:
                action = self.ai_explore.get_secure_uniform_action(self.last_state)
            else:
                action = self.rng.randint(self.env.nb_actions)
        else:  # safe greedy action
            action = self.ai.get_max_action(states=self.last_state, target=False)[0]
            if self.max_secure: # uses theorem 3 to also assert threshold 
                actions = self.ai_explore.get_secure_actions(self.last_state, target=False, q_threshold=self.q_threshold)
                if action not in actions and len(actions) > 0:
                    action = self.rng.choice(actions)  # prevents unsafe greedy actions

        new_obs, reward, game_over, _ = self.env.step(action)
        if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
            new_obs = new_obs.reshape(self.env.state_shape)
        if self.env.get_lives() <= 1:  # to avoid issues with the last life
            game_over = True
        # NOTE: `game_over` controls when episode is done,
        #       `explore/exploit_term` specify respectively terminal states for explore and exploit AI's.
        exploit_term = game_over
        explore_term = False
        explore_reward = 0.0
        if self.env.get_lives() < prev_lives or reward < 0:
            explore_reward = -1.0
            explore_term = True
        if reward > 0:
            self.goal_ += 1
        if reward > 0 or explore_reward < 0:
            temp_last_state = self.last_state.copy()
        if not evaluate:
            self.ai.transitions.add(s=self.last_state[-1].astype('float32'), a=action, r=reward, t=exploit_term)
            if self.secure and self.exploration_learning:
                self.ai_explore.transitions.add(s=self.last_state[-1].astype('float32'), a=action, r=explore_reward,
                                                t=explore_term)
            if self.annealing and self.total_training_steps >= self.replay_min_size:
                self.anneal_eps()
            self.total_training_steps += 1
        self._update_state(new_obs)
        if reward > 0 and not evaluate:
            if len(self.ai.rewarding_transitions) > self.ai_rewarding_buffer_size:
                self.ai.rewarding_transitions.pop(0)
            self.ai.rewarding_transitions.append((temp_last_state.astype('float32'), action, reward,
                                                  self.last_state.astype('float32'), exploit_term))
        if self.secure == True and explore_reward < 0 and not evaluate:
            if len(self.ai_explore.rewarding_transitions) > self.ai_explore_rewarding_buffer_size:
                self.ai_explore.rewarding_transitions.pop(0)
            self.ai_explore.rewarding_transitions.append((temp_last_state.astype('float32'), action, explore_reward,
                                                          self.last_state.astype('float32'), explore_term))
        return reward, game_over

    def _reset(self):
        self.last_episode_steps = 0
        self.score_agent = 0
        self.score_computer = 0

        assert self.max_start_nullops >= self.history_len or self.max_start_nullops == 0
        if self.max_start_nullops != 0:
            num_nullops = self.rng.randint(self.history_len, self.max_start_nullops)
            for i in range(num_nullops - self.history_len):
                self.env.step(0)

        for i in range(self.history_len):
            if i > 0:
                self.env.step(0)
            obs = self.env.get_state()
            if obs.ndim == 1 and len(self.env.state_shape) == 2:
                obs = obs.reshape(self.env.state_shape)
            self.last_state[i] = obs

    def _update_state(self, new_obs):
        temp_buffer = np.empty(self.last_state.shape, dtype=np.uint8)
        temp_buffer[:-1] = self.last_state[-self.history_len + 1:]
        temp_buffer[-1] = new_obs
        self.last_state = temp_buffer

    def anneal_eps(self):
        if self.total_training_steps > self.annealing_start:
            step = self.total_training_steps - self.annealing_start
            if self.epsilon > self.final_epsilon:
                decay = (self.start_epsilon - self.final_epsilon) * step / self.annealing_steps
                self.epsilon = self.start_epsilon - decay
            if step >= self.annealing_steps:
                self.epsilon = self.final_epsilon

    @staticmethod
    def _plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                        moving_average=False):
        for key in plot_dict:
            plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
                 kind=kind, legend=legend, index_col=None, moving_average=moving_average)
            write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

    @staticmethod
    def _create_folder(folder_location, folder_name):
        i = 0
        while os.path.exists(folder_location + folder_name + "%s" % i):
            i += 1
        folder_name = folder_location + folder_name + "%s" % i
        os.makedirs(folder_name)
        return folder_name

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
