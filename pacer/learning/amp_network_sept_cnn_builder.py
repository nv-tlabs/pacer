# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class AMPSeptCNNBuilder(AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPSeptCNNBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.topk = 5
            kwargs['input_shape'] = (kwargs['self_obs_size'] + params['task_mlp']["units"][-1]  + 20, ) # ZL: jank self_obs + CNN embedding size + +trajs

            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var


            self._build_task_cnn()

        def load(self, params):
            super().load(params)
            self._task_units = params['task_mlp']['units']
            self._task_activation = params['task_mlp']['activation']
            self._task_initializer = params['task_mlp']['initializer']
            self._task_cnn = params['task_cnn']
            return

        def eval_task(self, task_obs):
            traj_size = self.task_obs_size_detail['traj']
            task_obs_traj = task_obs[:, :traj_size]
            task_obs_cnn = task_obs[:, traj_size:]

            B, F = task_obs_cnn.shape
            if "heightmap" in self.task_obs_size_detail:
                image_res = np.sqrt(F).astype(int)
                task_obs_cnn = task_obs_cnn.view(B, image_res, image_res, 1)
            elif 'heightmap_velocity' in self.task_obs_size_detail:
                image_res = np.sqrt(F/3).astype(int)
                task_obs_cnn = task_obs_cnn.view(B, image_res, image_res, 3)

            task_cnn_feat = self._task_actor_cnn(task_obs_cnn.permute((0, 3, 1, 2))).reshape(B, -1)

            if image_res != 32:
                task_cnn_feat = self._cnn_mlp(task_cnn_feat)

            return torch.cat([task_cnn_feat, task_obs_traj], dim = -1)

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
            assert(obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here

            task_out = self.eval_task(task_obs)
            c_input = torch.cat([self_obs, task_out], dim = -1)

            c_out = self.critic_mlp(c_input)
            value = self.value_act(self.value(c_out))
            return value

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs) # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
            assert(obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here

            task_out = self.eval_task(task_obs)
            actor_input = torch.cat([self_obs, task_out], dim = -1)

            a_out = self.actor_mlp(actor_input)

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return


        def _build_task_cnn(self):
            task_obs_size, task_obs_size_detail = self.task_obs_size, self.task_obs_size_detail

            if "heightmap" in task_obs_size_detail:
                F = task_obs_size_detail["heightmap"]
                image_res = np.sqrt(F).astype(int)
                input_shape = [image_res, image_res, 1]
            elif "heightmap_velocity" in task_obs_size_detail:
                F = task_obs_size_detail["heightmap_velocity"]
                image_res = np.sqrt(F/3).astype(int)
                input_shape = [image_res, image_res, 3]

            input_shape = torch_ext.shape_whc_to_cwh(input_shape)
            cnn_args = {
                'ctype' : self._task_cnn['type'],
                'input_shape' : input_shape,
                'convs' :self._task_cnn['convs'],
                'activation' : self._task_cnn['activation'],
                'norm_func_name' : self.normalization,
            }

            # CNN Debug
            # self._task_cnn['convs'][0]['kernel_size'] = 4; self._task_cnn['convs'][0]['strides'] = 2; self._task_cnn['convs'][1]['kernel_size'] = 4; self._task_cnn['convs'][1]['strides'] = 2; self._task_cnn['convs'][2]['kernel_size'] = 4; self._task_cnn['convs'][2]['strides'] = 2
            # self._task_cnn['convs'][0]['padding'] = 1; self._task_cnn['convs'][1]['padding'] = 1; self._task_cnn['convs'][2]['padding'] = 1
            # self._task_actor_cnn = self._build_conv(**cnn_args)
            # test_input = torch.zeros([128, 1, 32, 32])
            # import ipdb
            # ipdb.set_trace()
            # self._task_actor_cnn(test_input)


            self._task_actor_cnn = self._build_conv(**cnn_args)

            if image_res != 32: # ZL: ungly code to figure out the right dimension
                mlp_args = {
                    'input_size': int((image_res/32) ** 2 * 256),
                    'units': [512, 256],
                    'activation': self._disc_activation,
                    'dense_func': torch.nn.Linear
                }
                self._cnn_mlp = self._build_mlp(**mlp_args)


            if self.separate:
                self._task_critic_cnn = self._build_conv(**cnn_args)

            return
