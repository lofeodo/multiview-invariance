# multiview-invariance

A toolkit for evaluating **VLM cross-viewpoint spatial reasoning invariance** using ScanNet 3D scenes.

The core idea: given a reconstructed 3D scene, analytically find camera placements where the apparent spatial relation between two objects (e.g. "A is left of B") flips between viewpoints. Rendered image pairs/triplets are then used to probe whether vision-language models give consistent spatial relation judgments across viewpoints.

An optional **reference arrow** can be placed in the scene pointing toward the midpoint between the two highlighted objects, serving as an unambiguous spatial anchor for the pair. The arrow's own viewpoint is recorded and can be rendered as an additional image.

---

## Quick start

```bash
python download_scenes.py --scenes 0
python generate_viewpoint_pairs.py --fov 60 --resolution_w 1024 --resolution_h 768 --max_pairs_per_scene 6 --min_object_volume 0.2 --reference-object --print-reference-image --scene_dir scannet_data/scene0000_00 --occlusion_ray_threshold 0.5
```

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

Download specific scenes by ID, a range starting from scene 0, or all scenes.

```bash
# Download a specific set of scenes
python download_scenes.py --scenes 0 1 2

# Download the first N scenes (scene0000_00 through scene000N-1_00)
python download_scenes.py --upto 10

# Download all scenes
python download_scenes.py
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

### 3. `generate_viewpoint_pairs.py` — Generate viewpoint pairs (main script)

The main pipeline. For each scene it:

1. Loads the mesh and instance annotations
2. Applies the axis-alignment matrix so the floor is flat and the up axis is detected automatically
3. Filters out structural elements (walls, floor, ceiling, etc.) and tiny objects
4. Enumerates all object pairs within a configurable distance range
5. Finds candidate viewpoints for each pair irrespective of any arrow position
6. For each pair, searches for a valid arrow position (visible from ≥ 2 viewpoints, not occluded, not embedded in geometry, at a safe minimum distance from each highlighted object)
7. If no valid arrow position is found, the pair is skipped
8. Validates each camera (geometry proximity, per-object occlusion, frustum coverage, projected size)
9. Renders images with the two target objects highlighted in distinct colors against a grayscale background, with the reference arrow drawn as a flat indicator pointing toward the pair midpoint
10. Saves rendered images and a full metadata JSON

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
| `--full-colour` | off | Render in original scene colours; disables grayscale conversion and object highlighting |
| `--reference-object` | off | Place a coloured arrow in each scene pointing toward the midpoint between the two highlighted objects |
| `--print-reference-image` | off | Render an extra image from the arrow's own viewpoint and save it as `objA_x_objB_y_view_arrow.png`. Requires `--reference-object` |
| `--max-arrow-occlusion` | `0.8` | Minimum fraction of arrow sample rays that must reach the arrow unblocked from a viewpoint (0–1). Used to determine whether the arrow is visible from each camera |
| `--seed` | `42` | Random seed (controls pair enumeration order) |
| `--log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

**Default skipped labels:** `wall floor ceiling window door doorframe doorpane curtain blinds windowsill beam column pipe stair railing floor mat`

To override the skip list entirely:
```bash
python generate_viewpoint_pairs.py --scene_dir scannet_data/scene0000_00 \
    --skip_labels wall floor ceiling
```

---

## Reference arrow

When `--reference-object` is passed, the pipeline places a flat coloured arrow in each accepted scene. The arrow:

- Points toward the midpoint between the two highlighted objects' centroids
- Is rendered as a flat indicator lying in the horizontal plane (flat face perpendicular to the world up axis, so a horizontal arrow looks like a flat road sign)
- Is placed at a fixed height above the floor (`0.7 m` by default)
- Must be visible (≥ `--max-arrow-occlusion`, default 80%) from at least 2 of the selected viewpoints, verified via open-space raycasting from each camera to sample points along the arrow shaft
- Must not be embedded inside any scene object (AABB check) or clipping into the scene mesh
- Must be at a safe minimum distance from each highlighted object: the arrow centroid must be at least as far from object X's centroid as the longest bounding-box dimension of object X (checked for both A and B)
- Both highlighted objects must also be sufficiently visible from the arrow's own camera pose, checked with the same raycasting occlusion logic used for regular viewpoints

The arrow's camera pose is defined as:

| Axis | Direction |
|---|---|
| Forward | Arrow position → pair midpoint |
| Up | World up (floor normal) |
| Right | Cross(forward, up) |

If `--print-reference-image` is also passed, an image is rendered from this pose (without the arrow itself, to avoid being inside it) and saved alongside the viewpoint images.

Object pairs for which no valid arrow position can be found are skipped entirely.

---

## Outputs

```
outputs/
├── scene0000_00/
│   ├── images/
│   │   ├── objA_3_objB_7_view_0.png          # viewpoint 0
│   │   ├── objA_3_objB_7_view_1.png          # viewpoint 1 (relation flipped)
│   │   ├── objA_3_objB_7_view_2.png          # optional diagonal viewpoint
│   │   ├── objA_3_objB_7_view_arrow.png      # arrow viewpoint (--print-reference-image)
│   │   └── ...
│   └── metadata.json
└── viewpoints/
    └── scene0000_00/
        └── ...                               # same images, mirrored here
```

### Image format

By default, rendered images show the scene in **grayscale** with the two target objects highlighted in **distinct colors** (e.g. object A in yellow, object B in blue). A coloured arrow (e.g. orange) is overlaid when `--reference-object` is active. Colors are assigned per-pair from a rotating palette and recorded in the metadata.

When `--full-colour` is passed, images are rendered in the original scene colors with no object highlighting or grayscale conversion.

The arrow-view image (`_view_arrow.png`) does **not** include the arrow itself, so both highlighted objects are fully visible from the arrow's perspective.

### metadata.json structure

```json
{
  "scene_id": "scene0000_00",
  "axis_alignment_applied": true,
  "up_axis": "Y",
  "camera_conventions": "OpenCV (x-right, y-down, z-forward)",
  "viewpoint_groups": [
    {
      "pair_id": "3_7",
      "object_A": {
        "instance_id": 3,
        "label": "chair",
        "color": "yellow",
        "color_rgb": [1.0, 0.85, 0.0]
      },
      "object_B": {
        "instance_id": 7,
        "label": "table",
        "color": "blue",
        "color_rgb": [0.15, 0.45, 1.0]
      },
      "reference_object_arrow": {
        "color": "orange",
        "color_rgb": [1.0, 0.5, 0.0],
        "pose": {
          "position_world": [-0.50, 1.69, 0.67],
          "forward_world":  [0.82, -0.12, 0.56],
          "right_world":    [0.57,  0.00, -0.82],
          "up_world":       [0.10,  0.99,  0.07],
          "up_convention":  "world_up — camera up-axis is aligned with the floor normal (axis 1)",
          "w2c_matrix":     [["...4x4 row-major OpenCV extrinsic matrix..."]],
          "fov_degrees":    60.0,
          "image_resolution": [1024, 768]
        },
        "spatial_relations_from_arrow": {
          "A_left_of_B": false,
          "A_right_of_B": true,
          "A_in_front_of_B": true,
          "A_behind_B": false,
          "A_above_B": false,
          "A_below_B": false
        },
        "image_path": "images/objA_3_objB_7_view_arrow.png"
      },
      "viewpoints": [
        {
          "viewpoint_index": 0,
          "image_path": "images/objA_3_objB_7_view_0.png",
          "fov_degrees": 60,
          "image_resolution": [1024, 768],
          "spatial_relations": {
            "A_left_of_B": true,
            "A_right_of_B": false,
            "A_in_front_of_B": false,
            "A_behind_B": true,
            "A_above_B": false,
            "A_below_B": false
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

The `reference_object_arrow` block is only present when `--reference-object` is used. The `image_path` inside it is only present when `--print-reference-image` is also used.

### Spatial relation conventions

Each axis is represented by a complementary pair. Both members of a pair cannot be true simultaneously, but both can be false when the objects are too close together along that axis (within the dead zone).

| Field | Meaning when `true` |
|---|---|
| `A_left_of_B` | A appears more than 20 px to the left of B in the image (projected pixel x) |
| `A_right_of_B` | A appears more than 20 px to the right of B in the image (projected pixel x) |
| `A_in_front_of_B` | A is more than 0.1 m closer to the camera than B (camera-space depth) |
| `A_behind_B` | A is more than 0.1 m further from the camera than B (camera-space depth) |
| `A_above_B` | A's centroid is more than 0.1 m higher than B's **and** A's bounding-box bottom is more than 0.1 m higher than B's (world-space up axis) |
| `A_below_B` | A's centroid is more than 0.1 m lower than B's **and** A's bounding-box bottom is more than 0.1 m lower than B's (world-space up axis) |

Left/right uses projected pixel coordinates (image-plane x, including perspective division), matching what appears visually left/right in the rendered image. Above/below requires both the centroid and the bounding-box bottom of A to be strictly higher (or lower) than B's by at least 0.1 m, using world-space up-axis coordinates; this is independent of camera perspective. Front/behind uses camera-space depth.

The `flipped_relations` list tells you which of these flipped across the viewpoint pair, e.g. `["left_right"]` means view 0 has A-left-of-B and view 1 has A-right-of-B.

Spatial relations in `spatial_relations_from_arrow` follow the same conventions but are computed from the arrow's camera pose rather than from any of the regular viewpoints.

---

## Typical workflow

```bash
# 1. Download scenes (specific, range, or all)
python download_scenes.py --scenes 0 1 2  # specific
python download_scenes.py --upto 10       # first 10
python download_scenes.py                 # all

# 2. (Optional) visually inspect one
python view_scene.py --scene 0

# 3. Generate viewpoint pairs with reference arrows
python generate_viewpoint_pairs.py \
    --scene_dir scannet_data/scene0000_00 \
    --reference-object --print-reference-image

# 4. Batch-process everything downloaded
python generate_viewpoint_pairs.py \
    --scene_dir scannet_data --batch \
    --max_pairs_per_scene 30 \
    --reference-object --print-reference-image
```
