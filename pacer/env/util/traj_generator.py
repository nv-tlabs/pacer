# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import joblib
import random
from pacer.utils.flags import flags
from pacer.env.tasks.base_task import PORT, SERVER

class TrajGenerator():
    def __init__(self, num_envs, episode_dur, num_verts, device, dtheta_max,
                 speed_min, speed_max, accel_max, sharp_turn_prob):


        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._dtheta_max = dtheta_max
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._sharp_turn_prob = sharp_turn_prob

        self._verts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self._verts = self._verts_flat.view((num_envs, num_verts, 3))

        env_ids = torch.arange(self.get_num_envs(), dtype=np.int)

        # self.traj_data = joblib.load("data/traj/traj_data.pkl")
        self.heading = torch.zeros(num_envs, 1)
        return

    def reset(self, env_ids, init_pos):
        n = len(env_ids)
        if (n > 0):
            num_verts = self.get_num_verts()
            dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0 # Sample the angles at each waypoint
            dtheta *= self._dtheta_max * self._dt

            dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0) # Sharp Angles Angle
            sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
            sharp_mask = torch.bernoulli(sharp_probs) == 1.0
            dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

            dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0) # Heading


            dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
            dspeed *= self._accel_max * self._dt
            dspeed[:, 0] = (self._speed_max - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min # Speed

            speed = torch.zeros_like(dspeed)
            speed[:, 0] = dspeed[:, 0]
            for i in range(1, dspeed.shape[-1]):
                speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, self._speed_max)

            ################################################
            if flags.fixed_path:
                dtheta[:, :] = 0 # ZL: Hacking to make everything 0
                dtheta[0, 0] = 0 # ZL: Hacking to create collision
                if len(dtheta) > 1:
                    dtheta[1, 0] = -np.pi # ZL: Hacking to create collision
                speed[:] = (self._speed_min + self._speed_max)/2
            ################################################

            if flags.slow:
                speed[:] = speed/4

            dtheta = torch.cumsum(dtheta, dim=-1)
            
            # speed[:] = 6
            seg_len = speed * self._dt

            dpos = torch.stack([torch.cos(dtheta), -torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
            dpos *= seg_len.unsqueeze(-1)
            dpos[..., 0, 0:2] += init_pos[..., 0:2]
            vert_pos = torch.cumsum(dpos, dim=-2)

            self._verts[env_ids, 0, 0:2] = init_pos[..., 0:2]
            self._verts[env_ids, 1:] = vert_pos

            ####### ZL: Loading random real-world trajectories #######
            if flags.real_path:
                rids = random.sample(self.traj_data.keys(), n)
                traj = torch.stack([
                    torch.from_numpy(
                        self.traj_data[id]['coord_dense'])[:num_verts]
                    for id in rids
                ],
                                   dim=0).to(self._device).float()

                traj[..., 0:2] = traj[..., 0:2] - (traj[..., 0, 0:2] - init_pos[..., 0:2])[:, None]
                self._verts[env_ids] = traj

        return

    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(
            f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')

        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis = -1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis = -2)
        self._verts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self._verts[env_ids]


    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)

                disc_grad = torch.autograd.grad(
                    new_val,
                    new_obs,
                    grad_outputs=torch.ones_like(new_val),
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True)

        return pos
