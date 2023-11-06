"""Example script demonstrating the basic ScenePic functionality."""
import os
import argparse

import numpy as np
import scenepic as sp
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import torch
import cv2
import joblib
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree,SkeletonMotion, SkeletonState
from uhc.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]


Name = "getting_started"
Title = "Getting Started"


data_dir = "data/smpl"
smpl_parser_n = SMPL_Parser(model_path=data_dir,gender="neutral")
smpl_parser_m = SMPL_Parser(model_path=data_dir,gender="male")
smpl_parser_f = SMPL_Parser(model_path=data_dir,gender="female")

# texture_path ="/hdd/zen/data/SURREAL/smpl_data/"
# texture_image = cv2.imread("/hdd/zen/data/SURREAL/smpl_data/textures/male/nongrey_male_0550.jpg")
pkl_dir = "output/renderings/smpl_ego_4_1-2022-10-05-20:51:40.pkl"
Name = pkl_dir.split("/")[-1].split(".")[0]
pkl_data = joblib.load(pkl_dir)
mujoco_2_smpl = [
        mujoco_joint_names.index(q) for q in joint_names
        if q in mujoco_joint_names
    ]

def build_scene() -> sp.Scene:
    scene = sp.Scene()
    scene.framerate = 30
    base_size = 600
    num_per_row = 4

    items = list(pkl_data.items())
    # num_items = 4
    num_items = len(items)
    for entry_key, data_seq in items[:num_items]:
        main = scene.create_canvas_3d(width=base_size,
                                      height=base_size,
                                      canvas_id=entry_key)
        gender, beta = data_seq['betas'][0], data_seq['betas'][1:]
        if gender == 0:
            smpl_parser = smpl_parser_n
            humanoid_color = np.array([[0, 1, 100]]).repeat(6890, axis=0)
        elif gender == 1:
            smpl_parser = smpl_parser_m
            humanoid_color = np.array([[0, 0.75, 1]]).repeat(6890, axis=0)
        else:
            smpl_parser = smpl_parser_f
            humanoid_color = np.array([[0.8, 0.15, 0.15]]).repeat(6890, axis=0)

        ground = scene.create_mesh("ground")
        ground.add_quad(color=sp.Colors.Gray,
                        p0=np.array([-50, -50, 0]),
                        p1=np.array([50, -50, 0]),
                        p2=np.array([50, 50, 0]),
                        p3=np.array([-50, 50, 0]),
                        normal =np.array([0, 0, 1]) )

        ref_jt_pos_full = data_seq['body_pos_full'].numpy()[::2]
        skeleon_motion = SkeletonMotion.from_dict(data_seq)
        offset = skeleon_motion.skeleton_tree.local_translation[0]
        global_rot = skeleon_motion.global_rotation
        B, J, N = global_rot.shape
        pose_quat_global = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)[::2] # downsample to 30 fps
        B_down = pose_quat_global.shape[0]
        body_trans = skeleon_motion.global_translation[::2]
        root_trans = body_trans[:, 0]
        root_trans_offset = root_trans - offset

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleon_motion.skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans,
            is_local=False)
        local_rot = new_sk_state.local_rotation
        pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(B_down, -1, 3)
        pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(B_down, -1)

        look_at_pos = root_trans[0].numpy()
        main_camera = sp.Camera(up_dir = np.array([0, 0, 1]), look_at = look_at_pos, center = look_at_pos + np.array([0, 4, 0]))

        spheres = []
        for jt_name in mujoco_joint_names:

            sphere = scene.create_mesh(f"{entry_key}_sphere_{jt_name}", layer_id="main")
            if jt_name in ["Head", "L_Hand", "R_Hand"]:
                curr_color = sp.Colors.White
                ball_scale = 0.15
            else:
                curr_color = sp.Colors.Green
                ball_scale = 0.05
            sphere.add_sphere(color=curr_color, transform=sp.Transforms.scale(ball_scale), )
            spheres.append(sphere)


        with torch.no_grad():
            vertices, joints = smpl_parser.get_joints_verts(
                pose=torch.from_numpy(pose_aa),
                th_trans=root_trans_offset,
                th_betas=torch.from_numpy(beta[None, ]))
            vertices_np = vertices.numpy()
        main.set_layer_settings_({"main": sp.LayerSettings(opacity = 0.9)})

        smpl_mesh = scene.create_mesh(f"{entry_key}_smpl", layer_id="main")
        smpl_mesh.add_mesh_without_normals(vertices=vertices_np[0],
                                           triangles=smpl_parser.faces,
                                           colors=humanoid_color)

        # now we will iteratively create each frame of the animation.
        for i in tqdm(range(pose_aa.shape[0])):
            main_frame = main.create_frame()
            if np.linalg.norm(root_trans[i] - look_at_pos) > 1:
                look_at_pos += (root_trans[i].numpy() - look_at_pos) * 0.05
                main_camera = sp.Camera(up_dir=np.array([0, 0, 1]),
                                        look_at=look_at_pos,
                                        center=look_at_pos +
                                        np.array([0, 4, 0]))
            main_frame.camera = main_camera
            mesh_update = scene.update_mesh_positions(f"{entry_key}_smpl",
                                                       vertices_np[i])

            main_frame.add_mesh(mesh_update)
            main_frame.add_mesh(ground)

            for j, jt_name in enumerate(mujoco_joint_names):
                sphere = spheres[j]
                main_frame.add_mesh(sphere, transform=sp.Transforms.translate(ref_jt_pos_full[i, j]))
    scene.grid(width=f"{base_size * num_per_row}px",
               grid_template_rows=" ".join([f"{base_size}px"] * (num_items // num_per_row)),
               grid_template_columns=" ".join([f"{base_size}px"] * num_items))

    for i in range(num_items):
        scene.place(
            list(pkl_data.keys())[i], f"{i// num_per_row + 1}",
            f"{i % num_per_row + 1}")
    return scene


def _parse_args():
    parser = argparse.ArgumentParser(Title)
    parser.add_argument("--script",
                        action="store_true",
                        help="Whether to save the scenepic as a JS file")
    return parser.parse_args()


def _main():
    args = _parse_args()
    scene = build_scene()
    # The scene is complete, so we write it to a standalone file.
    if args.script:
        # If you have an existing HTML page you want to add a scenepic
        # to, then you can save the scenepic as a self-contained
        # Javascript file.
        scene.save_as_script("{}.js".format(Name), standalone=True)
    else:
        # However, ScenePic will also create a basic HTML wrapper
        # and embed the Javascript into the file directly so you
        # have a single file containing everything.
        scene.save_as_html("output/html/{}.html".format(Name), title=Title)


if __name__ == "__main__":
    _main()