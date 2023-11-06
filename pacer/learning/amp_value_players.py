import torch


from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.amp_players as amp_players
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class AMPPlayerContinuousValue(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)
        return


    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            # self._amp_debug(info)
            self._task_value_debug(info)
            

        return


    def _task_value_debug(self, info):
        obs = info['obs']
        amp_obs = info['amp_obs']
        task_value = self._eval_task_value(obs)
        amp_obs_single = amp_obs[0:1]

        critic_value = self._eval_critic(obs)
        disc_pred = self._eval_disc(amp_obs_single)
        amp_rewards = self._calc_amp_rewards(amp_obs_single)
        disc_reward = amp_rewards['disc_rewards']
        plot_all = torch.cat([critic_value, task_value])
        plotter_names = ("task_value", "task")
        self.live_plotter(plot_all.cpu().numpy(), plotter_names = plotter_names)
        return

    def _eval_task_value(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_task_value(input)

    def live_plotter(self, w, plotter_names,  identifier='', pause_time=0.00000001):
        matplotlib.use("Qt5agg")
        num_lines = len(w)
        if not hasattr(self, 'lines'):
            size = 100
            self.x_vec = np.linspace(0, 1, size + 1)[0:-1]
            self.y_vecs = [np.array([0] * len(self.x_vec)) for i in range(7)]
            self.lines = [[] for i in range(num_lines)]
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()

            self.fig = plt.figure(figsize=(1, 1))
            ax = self.fig.add_subplot(111)
            # create a variable for the line so we can later update it

            for i in range(num_lines):
                l, = ax.plot(self.x_vec, self.y_vecs[i], '-o', alpha=0.8)
                self.lines[i] = l

            # update plot label/title
            plt.ylabel('Values')

            plt.title('{}'.format(identifier))
            plt.ylim((-0.75, 1.5))
            plt.gca().legend(plotter_names)
            plt.show()

        for i in range(num_lines):
            # after the figure, axis, and line are created, we only need to update the y-data
            self.y_vecs[i][-1] = w[i]
            self.lines[i].set_ydata(self.y_vecs[i])
            # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
            self.y_vecs[i] = np.append(self.y_vecs[i][1:], 0.0)

        # plt.pause(pause_time)
        self.fig.canvas.start_event_loop(0.001)