import argparse
import open3d as o3d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene ID (e.g. 0 or scene0000_00)")
args = parser.parse_args()

scene_id = args.scene if args.scene.startswith("scene") else f"scene{str(args.scene).zfill(4)}_00"
scene_dir = Path("scannet_data") / scene_id
ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"

print(f"Loading {ply_path}...")
mesh = o3d.io.read_triangle_mesh(str(ply_path))
mesh.compute_vertex_normals()

print(f"  Vertices : {len(mesh.vertices):,}")
print(f"  Triangles: {len(mesh.triangles):,}")
print(f"  Has colors: {mesh.has_vertex_colors()}")

o3d.visualization.draw_geometries(
    [mesh],
    window_name=f"ScanNet {scene_id}",
    width=1280,
    height=720,
)
