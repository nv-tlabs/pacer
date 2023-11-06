import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import joblib
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm

mesh_name = "with_less_car"
data_mesh = joblib.load(f"data/mesh/{mesh_name}.pkl")

depth = data_mesh['heigthmap'][:, ::-1].T
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

    change_threshold = 0.2
    # change_threshold = 1

    curr_map[grad > change_threshold] = True
    curr_map[grad < -change_threshold] = False
    prev_grads[grad > change_threshold] = grad[grad > change_threshold]

    curr_map_acc_small.append(curr_map.copy())
    col_slice_prev = col_slice.copy()

curr_map_acc_small.append(curr_map.copy())
curr_map_acc_small = ~np.stack(curr_map_acc_small, axis=1)


curr_map_acc = np.logical_and(curr_map_acc_small, curr_map_acc_big)

curr_map_acc[-1000:,:] = False
curr_map_acc[:1000, :] = False
curr_map_acc[:,-1000:] = False
curr_map_acc[:, :1000] = False
curr_map_acc[2300:5890, 6600:9800] = False
curr_map_acc[2700:5500, 2700:5500] = False

import ipdb
ipdb.set_trace()

plt.figure(dpi=100); plt.imshow(curr_map_acc, cmap='gist_gray'); plt.show()
# plt.figure(dpi=100); plt.imshow(depth, cmap='gist_gray'); plt.show()
# plt.figure(dpi=100); plt.imshow(depth[4963:6375, 5435:6650]); plt.show()


# plt.imsave(f'data/mesh/{mesh_name}.png', depth)
# plt.imsave(f'data/mesh/{mesh_name}_walkable.png', curr_map_acc)



# plt.figure(dpi=100)
# plt.imshow(curr_map_acc)
# plt.show()


# import glob
# import os
# import sys
# import pdb
# import os.path as osp
# sys.path.append(os.getcwd())
# import numpy as np
# import matplotlib.pyplot as plt

# semantic = np.load("/home/zen/dev/nv/output/Viewport/semantic/53.npy")
# plt.imshow(semantic[15:713, 162:1056])
# plt.show()