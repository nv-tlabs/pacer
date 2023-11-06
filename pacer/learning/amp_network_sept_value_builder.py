from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_sept_builder import AMPSeptBuilder
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class AMPSeptValueBuilder(AMPSeptBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPSeptValueBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPSeptBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self._build_task_value_mlp()
            

        def load(self, params):
            super().load(params)
            self._value_units = params['value_mlp']['units']
            self._value_activation = params['value_mlp']['activation']
            self._value_initializer = params['value_mlp']['initializer']
            return

        def eval_task_value(self, obs):
            task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
            assert(obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here

            task_out = self.eval_task(task_obs)
            task_value_out = self._task_value_mlp(task_out)

            value = self._value_logits(task_value_out)
            return value

        def _build_task_value_mlp(self):
            task_obs_size, task_obs_size_detail = self.task_obs_size, self.task_obs_size_detail
            assert ("traj" in task_obs_size_detail and "heightmap" in task_obs_size_detail)
            mlp_input_shape = task_obs_size_detail['traj'] + task_obs_size_detail['heightmap'] # Traj, heightmap

            self._task_value_mlp = nn.Sequential()
            mlp_args = {
                'input_size': mlp_input_shape,
                'units': self._value_units,
                'activation': self._value_activation,
                'dense_func': torch.nn.Linear
            }
            self._task_mlp = self._build_mlp(**mlp_args)
            mlp_out_size = self._value_units[-1]
            self._value_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self._task_value_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if "people" in task_obs_size_detail:
                raise NotImplemented

            torch.nn.init.uniform_(self._value_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._value_logits.bias)

            return

