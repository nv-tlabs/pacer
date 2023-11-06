# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces
import numpy as np
import torch
from pacer.env.tasks.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython


class VecTaskCPUWrapper(VecTaskCPU):

    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations)
        return


class VecTaskGPUWrapper(VecTaskGPU):

    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)
        return


class VecTaskPythonWrapper(VecTaskPython):

    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)

        self._amp_obs_space = spaces.Box(np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf)
        return

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)