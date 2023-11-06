import torch.nn as nn
from learning.amp_models import ModelAMPContinuous
import torch


class ModelAMPContinuousSept(ModelAMPContinuous):
    def __init__(self, network):
        super().__init__(network)
        return
