# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from shutil import ExecError
import torch
import numpy as np

import env.tasks.humanoid_pedestrain_terrain as humanoid_pedestrain_terrain
from env.tasks.humanoid_pedestrain_terrain import compute_group_observation, compute_location_observations
from isaacgym import gymapi
from isaacgym.torch_utils import *
from env.tasks.humanoid import dof_to_obs
from env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from pacer.utils.flags import flags
from utils import torch_utils
from isaacgym import gymtorch
import joblib
from poselib.poselib.core.rotation3d import quat_inverse, quat_mul
from tqdm import tqdm

HACK_MOTION_SYNC = False


class HumanoidPedestrianTerrainHand(humanoid_pedestrain_terrain.HumanoidPedestrianTerrain):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._height_gen = torch.ones((self.num_envs, 2)).to(self.device) * 0.5
        self._hand_idx = self._build_key_body_ids_tensor(["L_Hand", "R_Hand"])
        self.reward_raw = torch.zeros((self.num_envs, 2)).to(self.device)


    def _draw_task(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        norm_states = self.get_head_pose()
        base_quat = norm_states[:, 3:7]
        if not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)
        heading_rot = torch_utils.calc_heading_quat(base_quat)

        points = quat_apply(
            heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
            self.height_points) + (norm_states[:, :3]).unsqueeze(1)

        if (not self.headless) and self.show_sensors:
            self._sensor_pos[:] = points
        # self._sensor_pos[..., 2] += 0.3
        # self._sensor_pos[..., 2] -= 5

        traj_samples = self._fetch_traj_samples()

        self._marker_pos[:] = traj_samples
        self._marker_pos[..., 2] = self._humanoid_root_states[..., 2:3]  # jp hack # ZL hack

        hand_pos = self._rigid_body_pos[..., self._hand_idx, :]

        self._marker_pos[..., -2:, :] = hand_pos
        self._marker_pos[..., -2:, 2] = self._height_gen + self._humanoid_root_states[..., 2:3]

        # self._marker_pos[..., 2] = 0.89
        # self._marker_pos[..., 2] = 0

        if (not self.headless) and self.show_sensors:
            comb_idx = torch.cat([self._sensor_actor_ids, self._marker_actor_ids])
        else:
            comb_idx = torch.cat([self._marker_actor_ids])


        if flags.show_traj:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(comb_idx), len(comb_idx))

            self.gym.clear_lines(self.viewer)

            for i, env_ptr in enumerate(self.envs):
                verts = self._traj_gen.get_traj_verts(i)
                verts[..., 2] = self._humanoid_root_states[i, 2]  # ZL Hack
                # verts[..., 2] = 0.89
                # verts[..., 2] = 0
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines,
                                curr_cols)

        return


    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):

            obs_size = 2 * self._num_traj_samples + 2 # target height

            if self.terrain_obs:
                if self.velocity_map:
                    obs_size += self.num_height_points * 3
                else:
                    obs_size += self.num_height_points

            if self._divide_group and self._group_obs:
                obs_size += 5 * 11 * 3

        return obs_size

    def get_self_obs_size(self):
        return self._num_self_obs
 

    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)

        return

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)

        root_pos = self._humanoid_root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

        self._height_gen[env_ids] = torch.rand((len(env_ids), 2)).to(self.device)
        return


    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_hand_height = self._height_gen
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_hand_height = self._height_gen[env_ids]
        obs = []
        traj_samples = self._fetch_traj_samples(env_ids)

        loc_obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)
        obs.append(loc_obs)
        obs.append(tar_hand_height)
        if self.terrain_obs:

            if self.terrain_obs_root == "head":
                head_pose = self.get_head_pose(env_ids=env_ids)
                self.measured_heights = self.get_heights(root_states=head_pose, env_ids=env_ids)
            else:
                self.measured_heights = self.get_heights(root_states=root_states, env_ids=env_ids)

            if self.cfg['env'].get("use_center_height", False):
                center_heights = self.get_center_heights(root_states=root_states, env_ids=env_ids)
                center_heights = center_heights.mean(dim=-1, keepdim=True)
                heights = torch.clip(center_heights - self.measured_heights, -3, 3.) * self.height_meas_scale  #
                # joblib.dump(self.measured_heights, "heights.pkl")

            else:
                heights = torch.clip(root_states[:, 2:3] - self.measured_heights, -3, 3.) * self.height_meas_scale  #

            obs.append(heights)

        if self._divide_group and self._group_obs:
            group_obs = compute_group_observation(self._rigid_body_pos, self._rigid_body_rot, self._rigid_body_vel, self.selected_group_jts, self._group_num_people, self._has_upright_start)
            # Group obs has to be computed as a whole. otherwise, the grouping breaks.
            if not (env_ids is None):
                group_obs = group_obs[env_ids]

            obs.append(group_obs)

        obs = torch.cat(obs, dim=-1)

        return obs

    def _compute_flip_task_obs(self, normal_task_obs, env_ids):

        # location_obs  20
        # Terrain obs: self.num_terrain_obs
        # group obs
        B, D = normal_task_obs.shape
        traj_samples_dim = 20
        hand_height_samples_dim = 2
        obs_acc = []
        normal_task_obs = normal_task_obs.clone()
        traj_samples = normal_task_obs[:, :traj_samples_dim].view(B, 10, 2)
        traj_samples[..., 1] *= -1
        obs_acc.append(traj_samples.view(B, -1))

        hand_height_samples = normal_task_obs[:, traj_samples_dim:traj_samples_dim + hand_height_samples_dim]
        hand_height_samples = hand_height_samples[:, [1, 0]]
        obs_acc.append(hand_height_samples)

        if self.terrain_obs:
            if self.velocity_map:
                height_samples = normal_task_obs[..., (traj_samples_dim + hand_height_samples_dim): hand_height_samples_dim + traj_samples_dim + self.num_height_points * 3]
                height_samples = height_samples.view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)), 3)
                height_samples[..., 0].flip(2)
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))
            else:
                height_samples = normal_task_obs[..., traj_samples_dim: traj_samples_dim + self.num_height_points].view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)))
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))

        obs = torch.cat(obs_acc, dim=1)

        if self._divide_group and self._group_obs:
            group_obs = normal_task_obs[..., traj_samples_dim + self.num_height_points: ].view(B, -1, 3)
            group_obs[..., 1] *= -1
            obs_acc.append(group_obs.view(B, -1))

        obs = torch.cat(obs_acc, dim=1)

        del obs_acc

        return obs



    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        hand_pos = self._rigid_body_pos[..., self._hand_idx, :]

        self.rew_buf[:], self.reward_raw[:] = compute_location_reward(
            root_pos, hand_pos, tar_pos, self._height_gen)
        
        return

@torch.jit.script
def compute_location_reward(root_pos, hand_pos, tar_pos, tar_height):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    pos_err_scale = 2.0
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    traj_pos_reward = torch.exp(-pos_err_scale * pos_err)

    height_err_scale = 6.0
    hand_height_relative = hand_pos[..., 2] - root_pos[..., None, 2]
    hand_height_diff = hand_height_relative - tar_height
    hand_height_err = torch.sum(hand_height_diff * hand_height_diff, dim=-1)
    tar_height_reward = torch.exp(-hand_height_err * height_err_scale)

    reward = 0.6 * traj_pos_reward + 0.4 * tar_height_reward
    reward_raw = torch.stack([traj_pos_reward, tar_height_reward], dim = -1)
    return reward, reward_raw