import torch.nn as nn
from learning.amp_sept_models import ModelAMPContinuousSept
import torch


class ModelAMPContinuousSeptValue(ModelAMPContinuousSept):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMPContinuousSeptValue.Network(net)

    class Network(ModelAMPContinuousSept.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            result = super().forward(input_dict)
            obs = input_dict['obs']
            
            ### ZL: evaluate the additional task value function here
            task_values = self.a2c_network.eval_task_value(obs)
            result['task_values'] = task_values


            return result

        