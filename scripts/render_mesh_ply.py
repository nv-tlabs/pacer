import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import joblib
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage

# mesh_data = dict(np.load("data/mesh/mesh_simplified_3.npz"))
# mesh_data = dict(np.load("data/mesh/intersection.npz"))
# mesh_name = "parking_with_cars"

mesh_name = "mesh-downtown-san-jose-mapaligned-cropped-bottom-part-global"
mesh_data = trimesh.load(f'data/mesh/{mesh_name}.ply')
vertices = np.array(mesh_data.vertices).astype(np.float32)
faces = np.array(mesh_data.faces)

max_vals = vertices.max(axis = 0) -  vertices.min(axis = 0)
xdiff = vertices[:, 0].max() - (max_vals / 2)[0]
ydiff = vertices[:, 1].max() - (max_vals/2)[1]

mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)


extent = (vertices.max(axis = 0) - vertices.min(axis = 0)).astype(int)
# compose scene
scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
camera = pyrender.OrthographicCamera(xmag=extent[0]/2, ymag=extent[1]/2, znear=0.1, zfar=1000)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

scene.add(mesh, pose=np.eye(4))
scene.add(light, pose=np.eye(4))

cam_pos = np.array([xdiff, ydiff, 20])
print("cam_pos", cam_pos)
scene.add(camera, pose=[[1, 0, 0, cam_pos[0]], [0, 1, 0, cam_pos[1]], [0, 0, 1, cam_pos[2]], [0, 0, 0, 1]])

# pyrender.Viewer(scene, use_raymond_lighting=True)
# render scene
size_mul = 10 # reslution 1/20
print("size: ", extent[0] * size_mul, extent[1] * size_mul)
r = pyrender.OffscreenRenderer(extent[0] * size_mul, extent[1] * size_mul)
color, depth = r.render(scene)

depth = -depth + cam_pos[-1]
depth[depth == cam_pos[2]] = 0; print("manual fix a few issues with holes??"); print("manual fix a few issues with holes??")
print("depth range:", depth.max(), depth.min())


x_scale = size_mul
y_scale = size_mul

print("Scale:", x_scale, y_scale)

max_diff = vertices[:, 2].max() - vertices[:, 2].min()

walkable_map = np.ones_like(depth).astype(bool)
walkable_map[depth == 0] = False
walkable_map[depth > max_diff/2 + vertices[:, 2].min()] = False
walkable_map = ndimage.binary_erosion(walkable_map,
                                       iterations=10).astype(int)

joblib.dump(
    {
        "vertices": vertices,
        "faces": faces,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "cam_pos": cam_pos,
        "walkable_map": walkable_map.T[:, ::-1].copy(),
        "heigthmap": depth.T[:, ::-1].copy(),
    }, f"data/mesh/{mesh_name}.pkl")


plt.imsave(f'data/mesh/{mesh_name}.png', depth)

plt.figure(dpi=100)
plt.imshow(walkable_map)
plt.show()

plt.figure(dpi=100)
plt.imshow(depth)
plt.show()
