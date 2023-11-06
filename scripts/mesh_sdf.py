import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import mesh_to_sdf
import numpy as np

# parking= "intersection"
obj_name= "parking"
mesh_data = trimesh.load(f'data/mesh/{obj_name}.obj')
np.savez(f"data/mesh/{obj_name}", vertices = np.array(mesh_data.vertices).astype(np.float32), faces = np.array(mesh_data.faces))

query_points = np.random.random([10, 3])
print(
    mesh_to_sdf.mesh_to_sdf(mesh_data,
                            query_points,
                            surface_point_method='scan',
                            sign_method='normal',
                            bounding_radius=None,
                            scan_count=100,
                            scan_resolution=400,
                            sample_point_count=10000000,
                            normal_sample_count=11))
