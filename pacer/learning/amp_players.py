import torch


from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.common_player as common_player

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._normalize_input = config['normalize_input']
        self._disc_reward_scale = config['disc_reward_scale']

        super().__init__(config)

        # self.env.task.update_value_func(self._eval_critic, self._eval_actor)

        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])

            if self._normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()

        return

    def _eval_critic(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_critic(input)

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)

        return

    def _eval_task_value(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_task_value(input)


    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
            if self.env.task.has_task:
                config['self_obs_size'] = self.env.task.get_self_obs_size()
                config['task_obs_size'] = self.env.task.get_task_obs_size()
                config['task_obs_size_detail'] = self.env.task.get_task_obs_size_detail()
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
            if self.env.task.has_task:
                config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
                config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()
                config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()

        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs_single = amp_obs[0:1]

            # left_to_right_index = [
            #     4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22,
            #     13, 14, 15, 16, 17
            # ]
            # action = self._eval_actor(info['obs'])[0]
            # flip_action = self._eval_actor(info['flip_obs'])[0]
            # flip_action = flip_action.view(-1, 23, 3)
            # flip_action[..., 0] *= -1
            # flip_action[..., 2] *= -1
            # flip_action[..., :] = flip_action[..., left_to_right_index, :]
            # print("flip diff", (flip_action.view(-1, 69) - action).norm(dim = 1))

            disc_pred = self._eval_disc(amp_obs_single)
            amp_rewards = self._calc_amp_rewards(amp_obs_single)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]

            # print("disc_pred: ", disc_pred, disc_reward)

            # if not "rewards" in self.__dict__:
            #     self.rewards = []
            # self.rewards.append(
            #     self._calc_amp_rewards(
            #         info['amp_obs'])['disc_rewards'].squeeze())
            # if len(self.rewards) > 500:
            #     print(torch.topk(torch.stack(self.rewards).mean(dim = 0), k=150, largest=False)[1])
            #     import ipdb; ipdb.set_trace()
            #     self.rewards = []

        # jp hack
        with torch.enable_grad():
            amp_obs_single = amp_obs[0:1]
            amp_obs_single.requires_grad_(True)
            proc_amp_obs = self._preproc_amp_obs(amp_obs_single)
            disc_pred = self.model.a2c_network.eval_disc(proc_amp_obs)
            disc_grad = torch.autograd.grad(
                disc_pred,
                proc_amp_obs,
                grad_outputs=torch.ones_like(disc_pred),
                create_graph=False,
                retain_graph=True,
                only_inputs=True)
        

        
        grad_vals = torch.mean(torch.abs(disc_grad[0]), dim=0)
        if not "grad_acc" in self.__dict__:
            self.grad_acc = []
            self.reward_acc = []

        self.grad_acc.append(grad_vals)
        self.reward_acc.append(info['reward_raw'])
        if len(self.grad_acc) > 298:
            
            import joblib
            joblib.dump(self.grad_acc, "grad_acc.pkl")

            print(torch.stack(self.reward_acc).mean(dim = 0))
            self.grad_acc = []
            self.reward_acc = []
            # import ipdb; ipdb.set_trace()
            print("Dumping Grad info!!!!")

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _eval_actor(self, input):
        input = self._preproc_input(input)
        return self.model.a2c_network.eval_actor(input)

    def _preproc_input(self, input):
        if self._normalize_input:
            input = self.running_mean_std(input)
        return input

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
