# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.util import gym_util
from env.tasks.humanoid_pedestrain_terrain import HumanoidPedestrianTerrain
from isaacgym.torch_utils import *

from utils import torch_utils
from pacer.utils.flags import flags


class HumanoidPedestrianTerrainGetup(HumanoidPedestrianTerrain):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self._recovery_episode_prob_tgt  = self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps_tgt = self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob_tgt = self._fall_init_prob = cfg["env"]["fallInitProb"]
        if flags.server_mode:
            self._recovery_episode_prob_tgt  = self._recovery_episode_prob = 1
            self._fall_init_prob_tgt  =  self._fall_init_prob = 0

        self._reset_fall_env_ids = []
        
        self.availalbe_fall_states = torch.zeros(cfg["env"]['numEnvs']).long().to(device_type)
        self.fall_id_assignments = torch.zeros(cfg["env"]['numEnvs']).long().to(device_type)


        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._generate_fall_states()

        return

    def update_getup_schedule(self, epoch_num, getup_udpate_epoch=5000):
        ## Need to add aneal
        if epoch_num > getup_udpate_epoch:
            self._recovery_episode_prob = self._recovery_episode_prob_tgt
            self._fall_init_prob = self._fall_init_prob_tgt
        else:
            self._recovery_episode_prob = 0
            self._fall_init_prob = 1


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_recovery_count()
        return

    def _generate_fall_states(self):
        max_steps = 150
        # max_steps = 50000

        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        root_states = self._initial_humanoid_root_states[env_ids].clone()


        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7]) ## Random root rotation
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)

        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)
        root_states[..., :2] = new_root_xy
        center_height = self.get_center_heights(root_states,
                                                env_ids=env_ids).mean(dim=-1)
        root_states[..., 2] += center_height
        self._humanoid_root_states[env_ids] = root_states

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # _dof_state: from the currently simulated states
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self._dof_state)),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)

        self._refresh_sim_tensors()

        self._fall_root_states = self._humanoid_root_states.clone()
        self._fall_root_states[:, 7:13] = 0
        self._fall_dof_pos = self._dof_pos.clone()
        self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self.availalbe_fall_states[:] = 0
        self.fall_id_assignments[:] = 0

        return

    def resample_motions(self):
        super().resample_motions()
        self._generate_fall_states()
        self.reset() # Reset here should not cause the model to have collopsing episode lengths

        return

    def _reset_actors(self, env_ids):
        self.availalbe_fall_states[self.fall_id_assignments[env_ids]] = 0 # Clear out the assignment counters for these ones
        num_envs = env_ids.shape[0] # For these enviorments
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1) # If the env is terminated
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask) # for those env that have failed, with prob, turns them into recovery envs (this is harnessing episodes that natraully creates fall states)

        # reset the recovery counter for these envs. These env has the 150 steps for recovery from fall
        recovery_ids = env_ids[recovery_mask]
        if (len(recovery_ids) > 0):
            self._reset_recovery_episode(recovery_ids) # These are bonus recovery episodes

        # For the rest of the envs (terminated and not set to recovery), with probability self._fall_init_prob, make them to fall state
        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        fall_probs = to_torch(np.array([self._fall_init_prob] * nonrecovery_ids.shape[0]), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if (len(fall_ids) > 0):
            self._reset_fall_episode(fall_ids) # these automatically have recovery counter set to 60

        # These envs, are the normal ones with ref state init.
        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if (len(nonfall_ids) > 0):
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        # assert(len(recovery_ids) + len(fall_ids) + len(nonfall_ids) == len(env_ids))
        # Recovery_ids: naturally terminated and fall states, turn into natural fall state and learn from
        # fall_ids: natural resets, reset to fall state
        # nonfall_ids: natural resets, reset to ref states.

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return

    def _reset_fall_episode(self, env_ids):
        # fall_state_ids = torch.randperm(self._fall_root_states.shape[0])[:env_ids.shape[0]]
        self.availalbe_fall_states[self.fall_id_assignments[env_ids]] = 0 # ZL: Clear out the assignment counters for these ones. Clean out self. Should not need to do this?
        available_fall_ids = (self.availalbe_fall_states == 0).nonzero()
        assert(available_fall_ids.shape[0] >= env_ids.shape[0])
        fall_state_ids = available_fall_ids[torch.randperm(available_fall_ids.shape[0])][:env_ids.shape[0]].squeeze(-1)
        self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids

        self.availalbe_fall_states[fall_state_ids] = 1
        self.fall_id_assignments[env_ids] = fall_state_ids
        return

    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)

        
        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if (len(self._reset_fall_env_ids) > 0):
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0

        self.progress_buf[is_recovery] -= 1 # ZL: do not advance progress buffer for these. 
        return