"""
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_motion_interactive, plot_skeleton_state
from pacer.utils.motion_lib import MotionLib


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


# asset_descriptors = [
#     AssetDesc("mjcf/amp_humanoid.xml", False),
# ]
asset_descriptors = [
    AssetDesc("mjcf/amp_humanoid_sword_shield.xml", False),
]

# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[{
        "name":
        "--asset_id",
        "type":
        int,
        "default":
        0,
        "help":
        "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
    }, {
        "name": "--speed_scale",
        "type": float,
        "default": 1.0,
        "help": "Animation speed scale"
    }, {
        "name": "--show_axis",
        "action": "store_true",
        "help": "Visualize DOF axis"
    }])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" %
          (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "pacer/data/assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = asset_descriptors[
#     args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 6
spacing = 0
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.0)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# Setup Motion
body_ids = []
key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
for body_name in key_body_names:
    body_id = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0],
                                               body_name)
    assert (body_id != -1)
    body_ids.append(body_id)

body_ids = np.array(body_ids)
# _dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
# _dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
_dof_body_ids = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16]
_dof_offsets = [
    0, 3, 6, 9, 10, 13, 16, 17, 20, 21, 24, 27, 28, 31
]
_dof_obs_size = 72
_num_actions = 28

# motion_file = "data/reallusion/RL_Avatar_Move_WalkSlowToFast_M_Motion.npy"
motion_file = "data/reallusion_sword_shield/RL_Avatar_Fall_SpinLeft_Motion.npy"
motion_lib = MotionLib(motion_file=motion_file,
                       dof_body_ids=_dof_body_ids,
                       dof_offsets=_dof_offsets,
                       key_body_ids=body_ids,
                       device=args.compute_device_id)

current_dof = 0
speeds = np.zeros(num_dofs)

time_step = 0
dt = 1 / 60
rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    motion_id = 0
    motion_len = motion_lib.get_motion_length(motion_id).item()
    motion_time = time_step % motion_len
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = motion_lib.get_motion_state(
        torch.tensor([motion_id]).to(args.compute_device_id),
        torch.tensor([motion_time]).to(args.compute_device_id))

    if args.show_axis:
        gym.clear_lines(viewer)

    root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel],
                            dim=-1).cpu().repeat(num_envs, 1)

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))

    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)

    # diff = (rigidbody_state[:, body_ids, :3].to(args.compute_device_id) -
    #         key_pos).norm(dim=-1).mean()
    # print(diff)

    dof_states['pos'] = dof_pos.cpu().numpy()
    speed = speeds[current_dof]

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states,
                                 gymapi.STATE_POS)

        if args.show_axis:
            # get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i],
                                                  current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    time_step += dt

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
