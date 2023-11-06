# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

import env.tasks.humanoid_traj as humanoid_traj
from isaacgym import gymapi


class HumanoidPedestrian(humanoid_traj.HumanoidTraj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 10
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

        mesh_data = np.load("data/mesh/mesh_simplified_3.npz")
        mesh_vertices = mesh_data["vertices"]
        mesh_triangles = mesh_data["faces"].astype(np.uint32)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = mesh_vertices.shape[0]
        tm_params.nb_triangles = mesh_triangles.shape[0]
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.plane_static_friction
        tm_params.dynamic_friction = self.plane_dynamic_friction
        tm_params.restitution = self.plane_restitution

        self.gym.add_triangle_mesh(self.sim, mesh_vertices.flatten(order='C'), mesh_triangles.flatten(order='C'), tm_params)

        return
