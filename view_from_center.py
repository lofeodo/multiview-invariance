import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene ID (e.g. 0 or scene0000_00)")
args = parser.parse_args()

scene_id = args.scene if args.scene.startswith("scene") else f"scene{str(args.scene).zfill(4)}_00"
ply_path = Path("scannet_data") / scene_id / f"{scene_id}_vh_clean_2.ply"

print(f"Loading {ply_path}...")
mesh = o3d.io.read_triangle_mesh(str(ply_path))
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)

# Camera position: XY centroid of the scene, at standing eye height (~1.5m above floor)
floor_z = vertices[:, 2].min()
xy_center = vertices[:, :2].mean(axis=0)
camera_pos = np.array([xy_center[0], xy_center[1], floor_z + 1.5])

# Arbitrary focus object: centroid of vertices above floor by >0.5m (likely furniture/objects)
above_floor = vertices[vertices[:, 2] > floor_z + 0.5]
focus_point = above_floor.mean(axis=0) if len(above_floor) > 0 else vertices.mean(axis=0)

# front = direction from focus_point toward camera (open3d convention)
front = camera_pos - focus_point
norm = np.linalg.norm(front)
if norm < 1e-6:
    front = np.array([0.0, -1.0, 0.0])  # fallback: look along -Y
else:
    front = front / norm

print(f"Camera position: {camera_pos.round(3)}")
print(f"Focus point:     {focus_point.round(3)}")
print(f"Front vector:    {front.round(3)}")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"ScanNet {scene_id} — center view", width=1280, height=720)
vis.add_geometry(mesh)

ctr = vis.get_view_control()
ctr.set_lookat(focus_point.tolist())
ctr.set_front(front.tolist())
ctr.set_up([0.0, 0.0, 1.0])
ctr.set_zoom(0.5)

vis.run()
vis.destroy_window()
