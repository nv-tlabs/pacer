# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from ast import Try
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from enum import Enum
from matplotlib.pyplot import flag
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs, remove_base_rot, dof_to_obs_smpl
from env.util import gym_util
from pacer.utils.motion_lib import MotionLib
from pacer.utils.motion_lib_smpl import MotionLib as MotionLibSMPL

from isaacgym.torch_utils import *
from utils import torch_utils

from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import gc
from pacer.utils.flags import flags

HACK_MOTION_SYNC = False
# HACK_MOTION_SYNC = True
HACK_CONSISTENCY_TEST = False
HACK_OUTPUT_MOTION = False
HACK_OUTPUT_MOTION_ALL = False


class HumanoidAMP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        # jp hack
        if (HACK_MOTION_SYNC or HACK_CONSISTENCY_TEST):
            control_freq_inv = cfg["env"]["controlFrequencyInv"]
            self._motion_sync_dt = control_freq_inv * sim_params.dt
            cfg["env"]["controlFrequencyInv"] = 1
            cfg["env"]["pdControl"] = False

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._amp_root_height_obs = cfg["env"].get("ampRootHeightObs", False)

        assert (self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps,
             self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

        data_dir = "data/smpl"
        self.smpl_parser_n = SMPL_Parser(model_path=data_dir,
                                         gender="neutral").to(self.device)
        self.smpl_parser_m = SMPL_Parser(model_path=data_dir,
                                         gender="male").to(self.device)
        self.smpl_parser_f = SMPL_Parser(model_path=data_dir,
                                         gender="female").to(self.device)

        self.start = True # camera flag
        self.ref_motion_cache = {}
        return

    def resample_motions(self):
        # self.gym.destroy_sim(self.sim)
        # del self.sim
        # if not self.headless:
        #     self.gym.destroy_viewer(self.viewer)
        # self.create_sim()
        # self.gym.prepare_sim(self.sim)
        # self.create_viewer()
        # self._setup_tensors()
        print("Partial solution, only resample motions...")
        self._motion_lib.load_motions(skeleton_trees = self.skeleton_trees, limb_weights = self.humanoid_limb_and_weights.cpu(), gender_betas = self.humanoid_betas.cpu()) # For now, only need to sample motions since there are only 400 hmanoids
        # self.reset()
        # torch.cuda.empty_cache()
        # gc.collect()

    def pre_physics_step(self, actions):
        if (HACK_MOTION_SYNC or HACK_CONSISTENCY_TEST):
            actions *= 0

        if flags.debug:
            actions *= 0

        super().pre_physics_step(actions)
        return

    def post_physics_step(self):
        super().post_physics_step();

        # jp hack
        if (HACK_CONSISTENCY_TEST):
            self._hack_consistency_test()
        elif (HACK_MOTION_SYNC):
            self._hack_motion_sync()

        if (HACK_OUTPUT_MOTION):
            self._hack_output_motion()

        self._update_hist_amp_obs()  # One step for the amp obs

        self._compute_amp_observations()


        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat  ## ZL: hooks for adding amp_obs for trianing

        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
        #     torch.tensor([0]).to(self.device),
        #     torch.tensor([1.5]).to(self.device))
        # print(np.abs(dof_vel.cpu().numpy()).mean())
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        # Creates the reference motion amp obs. For discrinminiator

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)

        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(
            self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(
            -1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # Compute observation for the motion starting point
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, self._num_amp_obs_steps])

        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)


        if self.smpl_humanoid:
            motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

            amp_obs_demo = build_amp_observations_smpl(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, smpl_params, limb_weights, self.dof_subset, self._local_root_obs,
                self._amp_root_height_obs, self._has_dof_subset, self._has_shape_obs_disc, self._has_limb_weight_obs_disc,
                self._has_upright_start)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
                motion_ids, motion_times)

            amp_obs_demo = build_amp_observations(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, self._local_root_obs, self._amp_root_height_obs,
                self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        # ZL:

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/ov_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 37 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 40 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/smpl_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + len(self._dof_names) * 3 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

            if not self._amp_root_height_obs: self._num_amp_obs_per_step -= 1

            if self._has_dof_subset:
                self._num_amp_obs_per_step -= (6 + 3) * int((len(self._dof_names) * 3 - len(self.dof_subset)) / 3)

            if self._has_shape_obs_disc: self._num_amp_obs_per_step += 11
            # if self._has_limb_weight_obs_disc: self._num_amp_obs_per_step += 23 + 24 if not self._masterfoot else  29 + 30# 23 + 24 (length + weight)
            if self._has_limb_weight_obs_disc: self._num_amp_obs_per_step += 10

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert (False)

        if (self._enable_hist_obs):
            self._num_self_obs += self._num_amp_obs_steps * self._num_amp_obs_per_step

        return

    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)
        if self.smpl_humanoid:
            self._motion_lib = MotionLibSMPL(
                motion_file=motion_file,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device, masterfoot_conifg=self._masterfoot_config)
            self._motion_lib.load_motions(skeleton_trees = self.skeleton_trees, gender_betas = self.humanoid_betas.cpu(), limb_weights = self.humanoid_limb_and_weights.cpu(), random_sample=not HACK_MOTION_SYNC)

        else:
            self._motion_lib = MotionLib(
                motion_file=motion_file,
                dof_body_ids=self._dof_body_ids,
                dof_offsets=self._dof_offsets,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device)

        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert (
                False
            ), "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init))
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[
            env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _sample_time(self, motion_ids):
        return self._motion_lib.sample_time(motion_ids)

    def _get_fixed_smpl_state_from_motionlib(self, motion_ids, motion_times, curr_gender_betas):
        # Used for intialization. Not used for sampling. Only used for AMP, not imitation.
        motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, _, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        with torch.no_grad():
            gender = curr_gender_betas[:, 0]
            betas = curr_gender_betas[:, 1:]
            B, _ = betas.shape

            genders_curr = gender == 2
            height_tolorance = 0.02
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_f.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 1
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_m.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 0
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_n.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel

    def _get_smpl_state_from_motionlib_cache(self, motion_ids, motion_times):
        ## Chace the motion
        if not "motion_ids" in self.ref_motion_cache:
            self.ref_motion_cache['motion_ids'] = motion_ids
            self.ref_motion_cache['motion_times'] = motion_times
        else:
            if len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + \
                    (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() > 0:
                self.ref_motion_cache['motion_ids'] = motion_ids
                self.ref_motion_cache['motion_times'] = motion_times
            else:
                return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state_smpl(motion_ids, motion_times)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidAMP.StateInit.Random
                or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (
                False
            ), "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init))

        if self.smpl_humanoid:
            curr_gender_betas = self.humanoid_betas[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_motionlib(
                motion_ids, motion_times, curr_gender_betas)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
                motion_ids, motion_times)
            rb_pos, rb_rot = None, None

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = self._sample_ref_state(env_ids)

        if flags.debug:
            root_pos[..., 2] += 0.5

        if flags.fixed:
            x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
            root_pos[:, 0], root_pos[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times
        self._sampled_motion_ids[env_ids] = motion_ids
        if flags.follow:
            self.start = True  ## Updating camera when reset
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs),
                             device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _compute_humanoid_obs(self, env_ids=None):
        obs = super()._compute_humanoid_obs(env_ids)

        if (self._enable_hist_obs):
            if (env_ids is None):
                hist_obs = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
            else:
                hist_obs = self._amp_obs_buf[env_ids].view(
                    -1, self.get_num_amp_obs())

            obs = torch.cat([obs, hist_obs], dim=-1)

        return obs

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids,
                                   self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, self._num_amp_obs_steps-1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        if self.smpl_humanoid:
            motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

            amp_obs_demo = build_amp_observations_smpl(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, smpl_params, limb_weights, self.dof_subset, self._local_root_obs,
                self._amp_root_height_obs, self._has_dof_subset,
                self._has_shape_obs_disc, self._has_limb_weight_obs_disc, self._has_upright_start)

        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
            amp_obs_demo = build_amp_observations(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, self._local_root_obs, self._amp_root_height_obs,
                self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self,
                       env_ids,
                       root_pos,
                       root_rot,
                       dof_pos,
                       root_vel,
                       root_ang_vel,
                       dof_vel,
                       rigid_body_pos=None,
                       rigid_body_rot=None):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos

        self._dof_vel[env_ids] = dof_vel
        if (not rigid_body_pos is None) and (not rigid_body_rot is None):
            self._rigid_body_pos[env_ids] = rigid_body_pos
            self._rigid_body_rot[env_ids] = rigid_body_rot

            self._rigid_body_vel[env_ids] = 0
            self._rigid_body_ang_vel[env_ids] = 0
            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()

        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self._state_reset_happened and "_reset_rb_pos" in self.__dict__:
            # ZL: Hack to get rigidbody pos and rot to be the correct values. Needs to be called after _set_env_state
            env_ids = self._reset_ref_env_ids
            if len(env_ids) > 0:
                self._rigid_body_pos[env_ids] = self._reset_rb_pos
                self._rigid_body_rot[env_ids] = self._reset_rb_rot
                self._rigid_body_vel[env_ids] = 0
                self._rigid_body_ang_vel[env_ids] = 0
                self._state_reset_happened = False

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
            except:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)].clone()
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(
                self._num_amp_obs_steps - 1)]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if self.smpl_humanoid and self.dof_subset is None:
            # ZL hack
            self._dof_pos[:, 9:12], self._dof_pos[:, 21:24], self._dof_pos[:, 51:54], self._dof_pos[:,66:69] = 0, 0, 0, 0
            self._dof_vel[:, 9:12], self._dof_vel[:, 21:24], self._dof_vel[:, 51:54], self._dof_vel[:,66:69] = 0, 0, 0, 0

        # if (key_body_pos[..., 2].mean(dim = -1) > 2).sum():
        #     self.humanoid_betas[torch.where((key_body_pos[..
        # ., 2].mean(dim = -1) > 2))].cpu().numpy()
        #     import ipdb; ipdb.set_trace()
        #     print('bugg')
        # if flags.debug:
        # print(torch.topk(self._dof_pos.abs().sum(dim=-1), 5))

        if (env_ids is None):
            if self.smpl_humanoid:
                self._curr_amp_obs_buf[:] = build_amp_observations_smpl(
                    self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                    self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:, 0, :],
                    self._dof_pos, self._dof_vel, key_body_pos, self.humanoid_betas, self.humanoid_limb_and_weights,
                    self.dof_subset, self._local_root_obs,
                    self._amp_root_height_obs, self._has_dof_subset,
                    self._has_shape_obs_disc, self._has_limb_weight_obs_disc, self._has_upright_start)

            else:
                self._curr_amp_obs_buf[:] = build_amp_observations(
                    self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:,
                                                                        0, :],
                    self._rigid_body_vel[:,
                                         0, :], self._rigid_body_ang_vel[:,
                                                                         0, :],
                    self._dof_pos, self._dof_vel, key_body_pos,
                    self._local_root_obs, self._amp_root_height_obs,
                    self._dof_obs_size, self._dof_offsets)
        else:
            if len(env_ids) == 0: return
            if self.smpl_humanoid:
                self._curr_amp_obs_buf[env_ids] = build_amp_observations_smpl(
                    self._rigid_body_pos[env_ids][:, 0, :],
                    self._rigid_body_rot[env_ids][:, 0, :],
                    self._rigid_body_vel[env_ids][:, 0, :],
                    self._rigid_body_ang_vel[env_ids][:, 0, :],
                    self._dof_pos[env_ids], self._dof_vel[env_ids],
                    key_body_pos[env_ids], self.humanoid_betas[env_ids], self.humanoid_limb_and_weights[env_ids] , self.dof_subset,
                    self._local_root_obs, self._amp_root_height_obs,
                    self._has_dof_subset, self._has_shape_obs_disc, self._has_limb_weight_obs_disc,
                    self._has_upright_start)
            else:
                self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                    self._rigid_body_pos[env_ids][:, 0, :],
                    self._rigid_body_rot[env_ids][:, 0, :],
                    self._rigid_body_vel[env_ids][:, 0, :],
                    self._rigid_body_ang_vel[env_ids][:, 0, :],
                    self._dof_pos[env_ids], self._dof_vel[env_ids],
                    key_body_pos[env_ids], self._local_root_obs,
                    self._amp_root_height_obs, self._dof_obs_size,
                    self._dof_offsets)
        return

    def _hack_motion_sync(self):
        if (not hasattr(self, "_hack_motion_time")):
            self._hack_motion_time = 0.0

        num_motions = self._motion_lib.num_motions()
        motion_ids = np.arange(self.num_envs, dtype=np.int)
        motion_ids = np.mod(motion_ids, num_motions)
        # motion_ids[:] = 2
        motion_times = torch.tensor([self._hack_motion_time] * self.num_envs,
                                    dtype=torch.float32,
                                    device=self.device)
        if self.smpl_humanoid:
            motion_res = self._get_smpl_state_from_motionlib_cache(motion_ids, motion_times)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]



            # betas = self.humanoid_betas[0:1, 1:]  # ZL Hack before real body variation kicks in
            # vertices, joints = self.smpl_parser_n.get_joints_verts(
            #     torch.cat([
            #         torch_utils.quat_to_exp_map(root_rot).to(dof_pos), dof_pos
            #     ],
            #               dim=-1), betas, root_pos)
            # offset = joints[:, 0] - root_pos
            # root_pos[...,-1] -= (vertices - offset[:, None])[..., -1].min(dim=-1).values
            # root_pos[...,-1] += 0.03 # ALways slightly above the ground to avoid issue

        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times)
            rb_pos, rb_rot = None, None

        env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot)

        self._reset_env_tensors(env_ids)
        motion_dur = self._motion_lib._motion_lengths[0]
        self._hack_motion_time = np.fmod(self._hack_motion_time + self._motion_sync_dt, motion_dur.cpu().numpy())

        ###################################################################### Debug for left and right flip
        # body_pos = self._rigid_body_pos; body_rot = self._rigid_body_rot; body_vel = self._rigid_body_vel; body_ang_vel = self._rigid_body_ang_vel
        # root_pos = body_pos[:, 0, :]
        # root_rot = body_rot[:, 0, :]
        # root_pos_expand = root_pos.unsqueeze(-2)
        # local_body_pos = body_pos - root_pos_expand
        # obs = self._compute_humanoid_obs(env_ids)
        # flip_obs = self._compute_flip_humanoid_obs(env_ids)
        # diff = obs[0] - flip_obs[1]
        # import ipdb
        # ipdb.set_trace()
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[self.viewing_env_idx,
                                                   0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        # if np.abs(cam_pos[2] - char_root_pos[2]) > 5:
        cam_pos[2] = char_root_pos[2] + 0.5
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[self.viewing_env_idx],
                                     new_cam_pos, new_cam_target)

        if flags.follow:
            self.start = True
        else:
            self.start = False

        if self.start:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return


    def _hack_consistency_test(self):
        if (not hasattr(self, "_hack_motion_time")):
            self._hack_motion_time = 0.0

        motion_ids = np.array([0] * self.num_envs, dtype=np.int)
        motion_times = np.array([self._hack_motion_time] * self.num_envs)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)

        env_ids = torch.arange(self.num_envs,
                               dtype=torch.long,
                               device=self.device)
        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        self._reset_env_tensors(env_ids)

        motion_dur = self._motion_lib._motion_lengths[0]
        self._hack_motion_time = np.fmod(self._hack_motion_time + self.dt,
                                         motion_dur)

        self._refresh_sim_tensors()

        sim_key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if self.smpl_humanoid:
            print("ZL NOT FIXED YET")
            sim_amp_obs = build_amp_observations_smpl(
                self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :], self._rigid_body_ang_vel[:,
                                                                        0, :],
                self._dof_pos, self._dof_vel, sim_key_body_pos,
                self._local_root_obs, self._amp_root_height_obs, self._dof_offsets)

            ref_amp_obs = build_amp_observations_smpl(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, self._local_root_obs, self._amp_root_height_obs,
                self._dof_offsets)
        else:
            sim_amp_obs = build_amp_observations(
                self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :], self._dof_pos,
                self._dof_vel, sim_key_body_pos, self._local_root_obs,
                self._amp_root_height_obs, self._dof_obs_size, self._dof_offsets)

            ref_amp_obs = build_amp_observations(
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
                key_pos, self._local_root_obs, self._amp_root_height_obs,
                self._dof_obs_size, self._dof_offsets)

        obs_diff = sim_amp_obs - ref_amp_obs
        obs_diff = torch.abs(obs_diff)
        obs_err = torch.max(obs_diff, dim=0)

        return

    def _hack_output_motion(self):
        fps = 1.0 / self.dt
        from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
        from poselib.poselib.visualization.common import plot_skeleton_motion_interactive

        if (not hasattr(self, '_output_motion_root_pos')):
            self._output_motion_root_pos = []
            self._output_motion_global_rot = []

        root_pos = self._humanoid_root_states[..., 0:3].cpu().numpy()
        self._output_motion_root_pos.append(root_pos)

        body_rot = self._rigid_body_rot.cpu().numpy()
        rot_mask = body_rot[..., -1] < 0
        body_rot[rot_mask] = -body_rot[rot_mask]
        self._output_motion_global_rot.append(body_rot)

        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_root_pos) > 1):
            output_root_pos = np.array(self._output_motion_root_pos)
            output_body_rot = np.array(self._output_motion_global_rot)
            output_root_pos = to_torch(output_root_pos, device='cpu')
            output_body_rot = to_torch(output_body_rot, device='cpu')

            skeleton_tree = self._motion_lib._motions[0].skeleton_tree

            if (HACK_OUTPUT_MOTION_ALL):
                num_envs = self.num_envs
            else:
                num_envs = 1

            for i in range(num_envs):
                curr_body_rot = output_body_rot[:, i, :]
                curr_root_pos = output_root_pos[:, i, :]
                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    curr_body_rot,
                    curr_root_pos,
                    is_local=False)
                sk_motion = SkeletonMotion.from_skeleton_state(sk_state,
                                                               fps=fps)

                output_file = 'output/record_char_motion{:04d}.npy'.format(i)
                sk_motion.to_file(output_file)

                #plot_skeleton_motion_interactive(sk_motion)

            self._output_motion_root_pos = []
            self._output_motion_global_rot = []

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos,
                           dof_vel, key_body_pos, local_root_obs,
                           root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat(
        (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel,
                     local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos),
                    dim=-1)
    return obs


@torch.jit.script
def build_amp_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel,
                                dof_pos, dof_vel, key_body_pos, smpl_params, limb_weight_params,
                                dof_subset, local_root_obs, root_height_obs,
                                has_dof_subset, has_shape_obs_disc, has_limb_weight_obs, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, bool) -> Tensor
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat(
        (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs: obs_list.append(root_h)
    obs_list += [
        root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel,
        flat_local_key_pos
    ]
    # 6 + 3 + 3 + 114 + 57 + 12
    if has_shape_obs_disc: obs_list.append(smpl_params[:, :-6])
    if has_limb_weight_obs: obs_list.append(limb_weight_params)
    obs = torch.cat(obs_list, dim=-1)
    return obs