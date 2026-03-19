# multiview-invariance

A toolkit for evaluating **VLM cross-viewpoint spatial reasoning invariance** using ScanNet 3D scenes.

The core idea: given a reconstructed 3D scene, analytically find camera placements where the apparent spatial relation between two objects (e.g. "A is left of B") flips between viewpoints. Rendered image pairs/triplets are then used to probe whether vision-language models give consistent spatial relation judgments across viewpoints.

---

## Setup

### Requirements

```
pip install open3d numpy Pillow pyvista huggingface_hub
```

| Package | Purpose |
|---|---|
| `open3d` | Mesh loading, raycasting (occlusion checks) |
| `numpy` | All geometry math |
| `Pillow` | Saving rendered images |
| `pyvista` | Headless rendering (Windows-compatible via VTK) |
| `huggingface_hub` | Downloading ScanNet scenes |

> **Windows note:** Open3D's built-in `OffscreenRenderer` requires EGL (Linux only). This repo uses PyVista instead, which works headlessly on Windows via VTK software rendering.

---

## Data

Scene data is downloaded from the `zahidpichen/scannet-dataset` Hugging Face dataset and stored under `scannet_data/`.

Each scene directory (e.g. `scannet_data/scene0000_00/`) contains:

| File | Contents |
|---|---|
| `scene0000_00_vh_clean_2.ply` | Reconstructed mesh with per-vertex RGB colors |
| `scene0000_00_vh_clean_2.labels.ply` | Same mesh with per-vertex semantic label IDs |
| `scene0000_00_vh_clean_2.0.010000.segs.json` | Maps each vertex index → segment ID |
| `scene0000_00.aggregation.json` | Maps object instances → segment lists + labels |
| `scene0000_00.txt` | Scene metadata including axis alignment matrix |

---

## Scripts

### 1. `download_scenes.py` — Download ScanNet scenes

Download specific scenes by ID or a range starting from scene 0.

```bash
# Download a specific set of scenes
python download_scenes.py --scenes 0 1 2

# Download the first N scenes (scene0000_00 through scene000N-1_00)
python download_scenes.py --upto 10
```

Files are saved to `./scannet_data/`.

---

### 2. `view_scene.py` — Interactive 3D viewer

Open an interactive Open3D window to inspect a scene mesh.

```bash
python view_scene.py --scene 0
python view_scene.py --scene scene0000_00
```

Uses the vertex-colored mesh (`_vh_clean_2.ply`). Requires a display.

---

### 3. `view_from_center.py` — View scene from scene center

Opens the same interactive viewer but sets the camera to an eye-level position at the XY centroid of the scene, looking at the furniture centroid.

```bash
python view_from_center.py --scene 0
python view_from_center.py --scene scene0000_00
```

Useful for a quick sanity-check of scene orientation. Requires a display.

---

### 4. `generate_viewpoint_pairs.py` — Generate viewpoint pairs (main script)

The main pipeline. For each scene it:

1. Loads the mesh and instance annotations
2. Applies the axis-alignment matrix (so Y is up, floor is XZ plane)
3. Filters out structural elements (walls, floor, ceiling, etc.) and tiny objects
4. Enumerates all object pairs within a configurable distance range
5. Analytically places cameras on opposite sides of the "flip plane" for each pair
6. Validates each camera (checks for geometry collisions, occlusion, frustum coverage, projected size)
7. Renders images with the two target objects highlighted in distinct colors against a grayscale background
8. Saves rendered images and a full metadata JSON

#### Single scene

```bash
python generate_viewpoint_pairs.py --scene_dir scannet_data/scene0000_00
```

#### Batch (all scenes under a directory)

```bash
python generate_viewpoint_pairs.py --scene_dir scannet_data --batch
```

#### All options

| Argument | Default | Description |
|---|---|---|
| `--scene_dir` | *(required)* | Path to a scene dir, or root dir when using `--batch` |
| `--output_dir` | `outputs` | Root directory for all outputs |
| `--batch` | off | Process all `scene*` subdirs under `--scene_dir` |
| `--min_object_volume` | `0.005` m³ | Skip objects smaller than this (removes clutter) |
| `--min_centroid_distance` | `0.5` m | Minimum distance between object pair centroids |
| `--max_centroid_distance` | `5.0` m | Maximum distance between object pair centroids |
| `--standoff_distance_factor` | `1.5` | Camera standoff = this × centroid distance |
| `--standoff_min` | `1.0` m | Minimum camera standoff distance |
| `--standoff_max` | `4.0` m | Maximum camera standoff distance |
| `--camera_height` | `1.5` m | Camera height above floor |
| `--fov` | `60` deg | Vertical field of view |
| `--resolution_w` | `1024` px | Render width |
| `--resolution_h` | `768` px | Render height |
| `--min_projected_size` | `50` px | Reject viewpoints where an object's 2D bbox is smaller than this |
| `--occlusion_ray_threshold` | `0.20` | Fraction of sample rays that must reach an object (else it's occluded) |
| `--max_pairs_per_scene` | `20` | Cap on how many pairs to save per scene |
| `--skip_labels` | *(see below)* | Space-separated object labels to ignore |
| `--near_geom_dist` | `0.3` m | Camera is invalid if closer than this to any geometry |
| `--seed` | `42` | Random seed (controls pair enumeration order) |
| `--log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

**Default skipped labels:** `wall floor ceiling window door doorframe doorpane curtain blinds windowsill beam column pipe stair railing floor mat`

To override the skip list entirely:
```bash
python generate_viewpoint_pairs.py --scene_dir scannet_data/scene0000_00 \
    --skip_labels wall floor ceiling
```

---

## Outputs

```
outputs/
├── scene0000_00/
│   ├── images/
│   │   ├── objA_3_objB_7_view_0.png   # viewpoint 0
│   │   ├── objA_3_objB_7_view_1.png   # viewpoint 1 (relation flipped)
│   │   ├── objA_3_objB_7_view_2.png   # optional diagonal viewpoint
│   │   └── ...
│   └── metadata.json
└── viewpoints/
    └── scene0000_00/
        └── ...                         # same images, mirrored here
```

### Image format

Rendered images show the scene in **grayscale** with the two target objects highlighted in **distinct colors** (e.g. object A in yellow, object B in blue). Colors are assigned per-pair from a rotating palette and recorded in the metadata.

### metadata.json structure

```json
{
  "scene_id": "scene0000_00",
  "axis_alignment_applied": true,
  "viewpoint_groups": [
    {
      "pair_id": "3_7",
      "object_A": {
        "instance_id": 3,
        "label": "chair",
        "color": "yellow",
        "centroid_world": [x, y, z],
        "bbox_min_world": [x, y, z],
        "bbox_max_world": [x, y, z]
      },
      "object_B": {
        "instance_id": 7,
        "label": "table",
        "color": "blue",
        "centroid_world": [x, y, z],
        "bbox_min_world": [x, y, z],
        "bbox_max_world": [x, y, z]
      },
      "viewpoints": [
        {
          "viewpoint_index": 0,
          "image_path": "images/objA_3_objB_7_view_0.png",
          "camera_position_world": [x, y, z],
          "camera_look_at_world": [x, y, z],
          "camera_up": [0, 1, 0],
          "camera_extrinsic_w2c": [[4x4 matrix]],
          "camera_intrinsic": [[3x3 matrix]],
          "fov_degrees": 60,
          "image_resolution": [1024, 768],
          "object_A_bbox_2d": [x1, y1, x2, y2],
          "object_B_bbox_2d": [x1, y1, x2, y2],
          "object_A_cam_coords": [x, y, z],
          "object_B_cam_coords": [x, y, z],
          "spatial_relations": {
            "A_left_of_B": true,
            "A_in_front_of_B": false,
            "A_above_B": false
          },
          "viewpoint_label": "perp_pos",
          "angular_sep_from_view0_deg": 0.0
        }
      ],
      "flipped_relations": ["left_right"],
      "viewpoint_angular_separation_degrees": 85.3
    }
  ]
}
```

### Spatial relation conventions

Camera space follows the **OpenCV convention**: x-right, y-down, z-forward into scene.

| Field | Meaning when `true` |
|---|---|
| `A_left_of_B` | A's camera-x coordinate is less than B's (A is to the left in the image) |
| `A_in_front_of_B` | A is closer to the camera than B (smaller z) |
| `A_above_B` | A has a smaller camera-y than B (higher in the image) |

The `flipped_relations` list tells you which of these flipped across the viewpoint pair, e.g. `["left_right"]` means view 0 has A-left-of-B and view 1 has A-right-of-B.

---

## Typical workflow

```bash
# 1. Download a few scenes
python download_scenes.py --scenes 0 1 2

# 2. (Optional) visually inspect one
python view_scene.py --scene 0

# 3. Generate viewpoint pairs
python generate_viewpoint_pairs.py --scene_dir scannet_data/scene0000_00

# 4. Batch-process everything downloaded
python generate_viewpoint_pairs.py --scene_dir scannet_data --batch --max_pairs_per_scene 30
```
