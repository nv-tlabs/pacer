import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import joblib
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm

# mesh_data = dict(np.load("data/mesh/mesh_simplified_3.npz"))
# mesh_data = dict(np.load("data/mesh/intersection.npz"))
# mesh_name = "parking_with_cars"
# mesh_name = "parking"
# mesh_name = "with_less_car"
# mesh_data = trimesh.load(f'/home/zen/dev/nv/usd_asset/{mesh_name}.obj')
mesh_name = "with_less_car"
mesh_data = trimesh.load(f'/home/zen/dev/nv/exports/{mesh_name}.obj')
vertices = np.array(mesh_data.vertices).astype(np.float32)
faces = np.array(mesh_data.faces)

max_vals = vertices.max(axis = 0) -  vertices.min(axis = 0)
vertices[:, 1] -= vertices[:, 1].max() - (max_vals/2)[1]
vertices[:, 0] -= vertices[:, 0].max() - (max_vals/2)[0]

mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

extent = (vertices.max(axis = 0) - vertices.min(axis = 0)).astype(int)
# compose scene
scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
camera = pyrender.OrthographicCamera(xmag=extent[0]/2, ymag=extent[1]/2, znear=0.1, zfar=1000)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

scene.add(mesh, pose=np.eye(4))
scene.add(light, pose=np.eye(4))

cam_pos = np.array([0, 0, 100])
scene.add(camera, pose=[[1, 0, 0, cam_pos[0]], [0, 1, 0, cam_pos[1]], [0, 0, 1, cam_pos[2]], [0, 0, 0, 1]])


# pyrender.Viewer(scene, use_raymond_lighting=True)
# render scene
size_mul = 20
print("size: ", extent[0] * size_mul, extent[1] * size_mul)
r = pyrender.OffscreenRenderer(extent[0] * size_mul, extent[1] * size_mul)
color, depth = r.render(scene)

depth = -depth + cam_pos[-1]
depth[depth == cam_pos[2]] = 0; print("manual fix a few issues with holes??"); print("manual fix a few issues with holes??")
print("depth range:", depth.max(), depth.min())

x_scale = size_mul
y_scale = size_mul

print("Scale:", x_scale, y_scale)

print("Scanning")
curr_map_acc_big = []
col_slice_prev = depth[:,0]
curr_map = np.zeros(depth.shape[0]).astype(bool)
prev_grads = np.zeros(depth.shape[0])
for i in tqdm(range(1, depth.shape[-1])):
    col_slice = depth[:,i]
    grad = col_slice - col_slice_prev

    change_threshold = 1.5
    # change_threshold = 1

    curr_map[grad > change_threshold] = True
    curr_map[grad < -change_threshold] = False
    prev_grads[grad > change_threshold] = grad[grad > change_threshold]

    curr_map_acc_big.append(curr_map.copy())
    col_slice_prev = col_slice.copy()

curr_map_acc_big.append(curr_map.copy())
curr_map_acc_big = ~np.stack(curr_map_acc_big, axis = 1)


curr_map_acc_small = []
col_slice_prev = depth[:,0]
curr_map = np.zeros(depth.shape[0]).astype(bool)
prev_grads = np.zeros(depth.shape[0])
for i in tqdm(range(1, depth.shape[-1])):
    col_slice = depth[:,i]
    grad = col_slice - col_slice_prev

    change_threshold = 0.5
    # change_threshold = 1

    curr_map[grad > change_threshold] = True
    curr_map[grad < -change_threshold] = False
    prev_grads[grad > change_threshold] = grad[grad > change_threshold]

    curr_map_acc_small.append(curr_map.copy())
    col_slice_prev = col_slice.copy()

curr_map_acc_small.append(curr_map.copy())
curr_map_acc_small = ~np.stack(curr_map_acc_small, axis=1)


curr_map_acc = np.logical_and(curr_map_acc_small, curr_map_acc_big)
# joblib.dump({
#             "vertices": vertices,
#             "faces": faces,
#              "x_scale": x_scale,
#              "y_scale": y_scale,
#              "cam_pos": cam_pos,
#              "walkable_map": curr_map_acc,
#              "heigthmap": depth},
#              f"data/mesh/{mesh_name}.pkl")
curr_map_acc[-2000:,:] = False
curr_map_acc[:1500, :] = False
curr_map_acc[:,-1500:] = False
curr_map_acc[:, :1500] = False
curr_map_acc[2300:5890, 6600:9800] = False
curr_map_acc[2700:5500, 2700:5500] = False


joblib.dump(
    {
        "vertices": vertices,
        "faces": faces,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "cam_pos": cam_pos,
        "walkable_map": curr_map_acc.T[:, ::-1].copy(),
        "heigthmap": depth.T[:, ::-1].copy(),
    }, f"data/mesh/{mesh_name}.pkl")


import ipdb
ipdb.set_trace()

plt.figure(dpi=100); plt.imshow(curr_map_acc, cmap='gray'); plt.show()
plt.figure(dpi=100); plt.imshow(depth, cmap='gray'); plt.show()
plt.figure(dpi=100); plt.imshow(depth[4963:6375, 5435:6650]); plt.show()


plt.imsave(f'data/mesh/{mesh_name}.png', depth)
plt.imsave(f'data/mesh/{mesh_name}_walkable.png', curr_map_acc)
print(cam_pos)




# plt.figure(dpi=100)
# plt.imshow(curr_map_acc)
# plt.show()
