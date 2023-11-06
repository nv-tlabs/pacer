# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import If
import numpy as np
import os
import yaml
from tqdm import tqdm

from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *
from pacer.utils import torch_utils
import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


def fix_trans_height(pose_aa, trans, curr_gender_betas, smpl_parsers):
    with torch.no_grad():
        gender = curr_gender_betas[0]
        betas = curr_gender_betas[1:]
        height_tolorance = 0.0
        vertices_curr, joints_curr = smpl_parsers[gender.item()].get_joints_verts(pose_aa, betas[None, ], trans)
        offset = joints_curr[:, 0] - trans
        diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance).min()
        vertices_curr[..., 2].max() - vertices_curr[..., 2].min()
        trans[..., -1] -= diff_fix
        return trans

def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, smpl_parsers, masterfoot_config, queue, pid):
    # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
    res = {}
    for f in range(len(motion_data_list)):
        assert (len(ids) == len(motion_data_list))
        curr_id = ids[f] # id for this datasample
        curr_file = motion_data_list[f]
        curr_gender_beta = gender_betas[f]
        trans = curr_file['root_trans_offset'].clone()
        pose_aa = torch.from_numpy(curr_file['pose_aa'])
        if fix_height:
            trans = fix_trans_height(pose_aa, trans, curr_gender_beta,
                                          smpl_parsers)

        pose_quat_global = curr_file['pose_quat_global']
        B, J, N = pose_quat_global.shape


        if not masterfoot_config is None:


            num_bodies = len(masterfoot_config['body_names'])
            pose_quat_holder = np.zeros([B, num_bodies, N])
            pose_quat_holder[..., -1] = 1
            pose_quat_holder[...,masterfoot_config['body_to_orig_without_toe'], :] \
                = pose_quat_global[..., masterfoot_config['orig_to_orig_without_toe'], :]

            pose_quat_holder[..., [
                masterfoot_config['body_names'].index(name)
                for name in ["L_Toe", "L_Toe_1", "L_Toe_1_1", "L_Toe_2"]
            ], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["L_Ankle"]], :]
            pose_quat_holder[..., [
                masterfoot_config['body_names'].index(name)
                for name in ["R_Toe", "R_Toe_1", "R_Toe_1_1", "R_Toe_2"]
            ], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["R_Ankle"]], :]

            pose_quat_global = pose_quat_holder

        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_trees[f],
            torch.from_numpy(pose_quat_global),
            trans,
            is_local=False)

        curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
        curr_dof_vels = compute_motion_dof_vels(curr_motion)

        curr_motion.dof_vels = curr_dof_vels
        curr_motion.gender_beta = curr_gender_beta
        res[curr_id] = (curr_file, curr_motion)

    if not queue is None:
        queue.put(res)
    else:
        return res

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, key_body_ids, device, fix_height = True, masterfoot_conifg = None, min_length = -1):
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._motion_data = joblib.load(motion_file)

        if min_length != -1:
            data_list = {k: v for k, v in list(self._motion_data.items()) if len(v['pose_quat_global']) >= min_length}
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(list(self._motion_data.values()))
            self._motion_data_keys = np.array(list(self._motion_data.keys()))
        self._num_unique_motions = len(self._motion_data_list)


        self._masterfoot_conifg = masterfoot_conifg
        data_dir = "data/smpl"
        smpl_parser_n = SMPL_Parser(model_path=data_dir,
                                         gender="neutral")
        smpl_parser_m = SMPL_Parser(model_path=data_dir,
                                         gender="male")
        smpl_parser_f = SMPL_Parser(model_path=data_dir,
                                         gender="female")
        self.smpl_parsers = {
            0: smpl_parser_n,
            1: smpl_parser_m,
            2: smpl_parser_f
        }
        self.fix_height = fix_height

        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions)
        self._success_rate = torch.zeros(self._num_unique_motions)
        self._sampling_history = torch.zeros(self._num_unique_motions)
        self._sampling_prob = torch.ones(self._num_unique_motions)/self._num_unique_motions # For use in sampling batches
        self._sampling_batch_prob = None # For use in sampling within batches
        return

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample = True, start_idx = 0):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del self.gts , self.grs , self.lrs, self.grvs, self.gravs , self.gavs , self.gvs, self.dvs,
            del  self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa , self._motion_quat

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []
        self._motion_quat = []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        num_motion_to_load = len(skeleton_trees)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob,num_samples=num_motion_to_load, replacement=True)
        else:
            sample_idxes = torch.clip(torch.arange(len(skeleton_trees)) + start_idx, 0, self._num_unique_motions - 1)

        self._curr_motion_ids = sample_idxes

        self._sampling_batch_prob =  self._sampling_prob[self._curr_motion_ids]/self._sampling_prob[self._curr_motion_ids].sum()

        print("Sampling motion:", sample_idxes[:10])

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        mp.set_sharing_strategy('file_system')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = mp.cpu_count()

        if num_jobs <= 8: num_jobs = 1
        # num_jobs = 1

        res_acc = {} # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk],
                 skeleton_trees[i:i + chunk], gender_betas[i:i + chunk],
                 self.fix_height, self.smpl_parsers, self._masterfoot_conifg)
                for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]

        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=load_motion_with_skeleton,
                                args=worker_args)
            worker.start()
        res_acc.update(load_motion_with_skeleton(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data['pose_aa'].reshape(-1, 72))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                print("no beta, using default value")
                self._motion_bodies.append(np.zeros(17))
                self._motion_quat.append(np.zeros(96))
                self._motion_aa.append(np.zeros(72))

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

        self._motion_lengths = torch.tensor(self._motion_lengths,
                                            device=self._device,
                                            dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps,
                                        device=self._device,
                                        dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(self._motion_aa),
                                       device=self._device,
                                       dtype=torch.float32)

        self._motion_quat = torch.tensor(self._motion_quat,
                                         device=self._device,
                                         dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt,
                                       device=self._device,
                                       dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames,
                                               device=self._device)
        self._motion_limb_weights = torch.tensor(limb_weights,
                                                device=self._device,
                                                dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float()
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions),
                                       dtype=torch.long,
                                       device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints


        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(
            num_motions, total_len))
        return

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def update_sampling_weight(self):
        sampling_temp = 0.2
        curr_termination_prob = 0.5
        curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids]/ self._sampling_history[self._curr_motion_ids]
        self._success_rate[self._curr_motion_ids] = curr_succ_rate
        sample_prob = torch.exp(-self._success_rate/sampling_temp)

        self._sampling_prob =  sample_prob/sample_prob.sum()
        self._termination_history[self._curr_motion_ids] = 0
        self._sampling_history[self._curr_motion_ids] = 0

        topk_sampled = self._sampling_prob.topk(50)
        print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])


    def update_sampling_history(self, env_ids):
        self._sampling_history[self._curr_motion_ids[env_ids]] += 1
        # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    def update_termination_history(self, termination):
        self._termination_history[self._curr_motion_ids] += termination.cpu()
        # print("termination history: ", self._termination_history[self._curr_motion_ids])


    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob,
                                       num_samples=n,
                                       replacement=True).to(self._device)

        return motion_ids


    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1/30
        motion_time = ((phase * motion_len)/curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids = None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids = None):
        if motion_ids is None:
            return self._motion_num_frames
        else:
            return self._motion_num_frames[motion_ids]

    def get_motion_state_smpl_interval(self, motion_ids, motion_times, offset = None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)

        frame_idx = torch.round((frame_idx0 * (1 - blend) + frame_idx1 * blend)).long()
        fl = frame_idx + self.length_starts[motion_ids]

        local_rot = self.lrs[fl]

        body_vel = self.gvs[fl]
        body_ang_vel = self.gavs[fl]

        rg_pos = self.gts[fl, :] # ZL: apply offset

        dof_vel = self.dvs[fl]

        vals = [local_rot, body_vel, body_ang_vel, rg_pos]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)
        key_pos = rg_pos[:, self._key_body_ids]

        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot = self.grs[fl]
        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "key_pos": key_pos,
            "motion_aa": self._motion_aa[fl],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_motion_state_smpl(self, motion_ids, motion_times, offset = None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]


        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [
             local_rot0, local_rot1, body_vel0, body_vel1,
                body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1   # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]   # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1

        key_pos = rg_pos[:, self._key_body_ids]

        local_rot = torch_utils.slerp(local_rot0, local_rot1,
                                      torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)

        # self.torch_humanoid.fk_batch()

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "key_pos": key_pos,
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]


        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1   # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0) # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

    # jp hack
    def _hack_test_vel_consistency(self, motion):
        test_vel = np.loadtxt("output/vel.txt", delimiter=",")
        test_root_vel = test_vel[:, :3]
        test_root_ang_vel = test_vel[:, 3:6]
        test_dof_vel = test_vel[:, 6:]

        dof_vel = motion.dof_vels
        dof_vel_err = test_dof_vel[:-1] - dof_vel[:-1]
        dof_vel_err = np.max(np.abs(dof_vel_err))

        root_vel = motion.global_root_velocity.numpy()
        root_vel_err = test_root_vel[:-1] - root_vel[:-1]
        root_vel_err = np.max(np.abs(root_vel_err))

        root_ang_vel = motion.global_root_angular_velocity.numpy()
        root_ang_vel_err = test_root_ang_vel[:-1] - root_ang_vel[:-1]
        root_ang_vel_err = np.max(np.abs(root_ang_vel_err))

        return