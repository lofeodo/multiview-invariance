import argparse
import open3d as o3d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene ID as a number (e.g. 211) or full name (e.g. scene0211_01)")
args = parser.parse_args()

if args.scene.startswith("scene"):
    scene_id = args.scene
else:
    # Numeric input — find all variants (scene0211_00, scene0211_01, …) that exist on disk
    prefix = f"scene{str(args.scene).zfill(4)}_"
    scannet = Path("scannet_data")
    matches = sorted(d.name for d in scannet.iterdir() if d.is_dir() and d.name.startswith(prefix))

    if not matches:
        print(f"No scene directories found matching '{prefix}*' under scannet_data/.")
        raise SystemExit(1)
    if len(matches) > 1:
        print(f"Ambiguous scene ID '{args.scene}' — multiple variants found:")
        for m in matches:
            print(f"  {m}")
        print("Re-run with the full scene name, e.g.: --scene", matches[0])
        raise SystemExit(1)

    scene_id = matches[0]

scene_dir = Path("scannet_data") / scene_id
ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"

if not ply_path.exists():
    print(f"Mesh file not found: {ply_path}")
    raise SystemExit(1)

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
