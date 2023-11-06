# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from env.tasks.humanoid import Humanoid
from env.tasks.humanoid_amp import HumanoidAMP
from env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from env.tasks.humanoid_traj import HumanoidTraj
from env.tasks.humanoid_pedestrian import HumanoidPedestrian
from env.tasks.humanoid_pedestrain_terrain import HumanoidPedestrianTerrain
from pacer.env.tasks.humanoid_pedestrain_terrain_hand import HumanoidPedestrianTerrainHand
from env.tasks.humanoid_pedestrain_terrain_getup import HumanoidPedestrianTerrainGetup
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper

from isaacgym import rlgpu

import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless)
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf))

    return task, env
