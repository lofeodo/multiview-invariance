"""
generate_viewpoint_pairs.py

Generate viewpoint pairs/triplets from ScanNet 3D scene meshes where the
apparent spatial relation (left/right, front/behind) between two objects
flips between viewpoints. Used to evaluate VLM cross-viewpoint spatial
reasoning invariance.

Coordinate conventions:
  - World space: axis-aligned so Y-up (after applying axisAlignment from
    scene .txt). Floor is approximately the XZ plane.
  - Camera space (OpenCV): x-right, y-down, z-forward into the scene.
  - Open3D extrinsic = world-to-camera (w2c) 4×4 matrix.

Usage:
    # Single scene
    python generate_viewpoint_pairs.py --scene_dir scannet_data/scene0000_00

    # Batch
    python generate_viewpoint_pairs.py --scene_dir scannet_data --batch
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SKIP_LABELS = {
    "wall", "floor", "ceiling", "window", "door",
    "doorframe", "doorpane", "curtain", "blinds",
    "windowsill", "beam", "column", "pipe", "stair",
    "railing", "floor mat",
}
DEFAULT_MIN_OBJECT_VOLUME = 0.005   # m³
DEFAULT_MIN_CENTROID_DIST = 0.5     # m
DEFAULT_MAX_CENTROID_DIST = 5.0     # m
DEFAULT_STANDOFF_FACTOR = 1.5       # ×(centroid distance)
DEFAULT_STANDOFF_MIN = 1.0          # m
DEFAULT_STANDOFF_MAX = 4.0          # m
DEFAULT_CAMERA_HEIGHT = 1.5         # m above floor
DEFAULT_FOV = 60.0                  # degrees
DEFAULT_RES_W = 1024
DEFAULT_RES_H = 768
DEFAULT_MIN_PROJ_SIZE = 50          # pixels
DEFAULT_OCCLUSION_THRESH = 0.20     # fraction of rays that must reach object
DEFAULT_MAX_PAIRS = 20              # per scene
DEFAULT_NEAR_GEOM_DIST = 0.3        # m — camera collision threshold

# ---------------------------------------------------------------------------
# Highlight colour palette
# ---------------------------------------------------------------------------
# Adjacent pairs are perceptually distinct so A and B never look similar.
# Stored as (name, float RGB in [0,1]).
HIGHLIGHT_PALETTE: list[tuple[str, np.ndarray]] = [
    ("yellow",  np.array([1.00, 0.85, 0.00])),
    ("blue",    np.array([0.15, 0.45, 1.00])),
    ("orange",  np.array([1.00, 0.50, 0.00])),
    ("cyan",    np.array([0.00, 0.85, 0.95])),
    ("red",     np.array([0.95, 0.15, 0.15])),
    ("lime",    np.array([0.50, 0.95, 0.10])),
    ("magenta", np.array([0.90, 0.10, 0.90])),
    ("teal",    np.array([0.10, 0.80, 0.65])),
]


def assign_pair_colors(
    pair_index: int,
) -> tuple[str, np.ndarray, str, np.ndarray, str, np.ndarray]:
    """Return (name_a, rgb_a, name_b, rgb_b, name_arrow, rgb_arrow) for a given pair index.

    Entries are drawn from HIGHLIGHT_PALETTE at offsets 0, 1, 2 (mod n) relative
    to the pair's base index, so A, B, and the arrow are always distinct colours.
    """
    n = len(HIGHLIGHT_PALETTE)
    name_a, rgb_a       = HIGHLIGHT_PALETTE[(pair_index * 2)     % n]
    name_b, rgb_b       = HIGHLIGHT_PALETTE[(pair_index * 2 + 1) % n]
    name_sp, rgb_sp     = HIGHLIGHT_PALETTE[(pair_index * 2 + 2) % n]
    return name_a, rgb_a.copy(), name_b, rgb_b.copy(), name_sp, rgb_sp.copy()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_axis_alignment(txt_path: Path) -> np.ndarray:
    """Parse axisAlignment from scene .txt, return 4×4 float64 matrix.
    Returns identity if the field is absent or the file does not exist."""
    mat = np.eye(4, dtype=np.float64)
    if not txt_path.exists():
        log.warning("Scene .txt not found: %s — assuming identity alignment", txt_path)
        return mat
    with txt_path.open() as fh:
        for line in fh:
            if line.startswith("axisAlignment"):
                vals = line.split("=", 1)[1].strip().split()
                mat = np.array([float(v) for v in vals], dtype=np.float64).reshape(4, 4)
                log.debug("axisAlignment matrix loaded from %s", txt_path)
                return mat
    log.debug("No axisAlignment in %s — using identity", txt_path)
    return mat


def load_mesh_and_instances(
    ply_path: Path,
    labels_ply_path: Path,
    agg_path: Path,
    segs_path: Path,
    axis_mat: np.ndarray,
    skip_labels: set[str],
    min_volume: float,
) -> tuple[o3d.geometry.TriangleMesh, list[dict]]:
    """Load mesh and compute per-instance data.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh — full scene mesh (axis-aligned, Y-up)
    instances : list of dicts with keys
        instance_id, label, centroid, bbox_min, bbox_max, vertex_indices
    """
    log.info("Loading mesh: %s", ply_path)
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)  # (N, 3) before alignment

    # Apply axis alignment
    N = vertices.shape[0]
    ones = np.ones((N, 1))
    verts_h = np.hstack([vertices, ones])          # (N, 4)
    verts_aligned = (axis_mat @ verts_h.T).T[:, :3]  # (N, 3)

    # Overwrite mesh vertices with aligned coordinates
    mesh.vertices = o3d.utility.Vector3dVector(verts_aligned)
    mesh.compute_vertex_normals()
    vertices = verts_aligned  # use aligned from here on

    # Load aggregation JSON
    with agg_path.open() as fh:
        agg = json.load(fh)

    # Load segmentation JSON
    with segs_path.open() as fh:
        segs_data = json.load(fh)
    seg_indices: list[int] = segs_data["segIndices"]  # parallel to vertex array
    seg_indices_arr = np.array(seg_indices, dtype=np.int32)

    instances: list[dict] = []
    for seg_group in agg["segGroups"]:
        label: str = seg_group.get("label", "unknown").lower().strip()
        if label in skip_labels:
            continue

        seg_ids = set(seg_group["segments"])
        # Find all vertex indices whose segment ID is in this instance
        vertex_mask = np.isin(seg_indices_arr, list(seg_ids))
        vert_idx = np.where(vertex_mask)[0]
        if len(vert_idx) == 0:
            continue

        pts = vertices[vert_idx]  # (k, 3)
        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        dims = bbox_max - bbox_min
        volume = float(dims[0] * dims[1] * dims[2])

        if volume < min_volume:
            continue

        instances.append(
            {
                "instance_id": int(seg_group["objectId"]),
                "label": label,
                "centroid": centroid,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "vertex_indices": vert_idx,
                "volume": volume,
            }
        )

    log.info("Loaded %d valid object instances", len(instances))
    return mesh, instances


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------

def look_at_matrix(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = None
) -> np.ndarray:
    """Compute OpenCV-convention world-to-camera (w2c) 4×4 matrix.

    Convention: x-right, y-down, z-forward (into scene).
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        # Degenerate: camera on top of target; use fallback
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # forward is parallel to up; pick another up
        alt_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up_corrected = np.cross(right, forward)  # reorthogonalize; points upward in cam

    # R maps world axes to camera axes.
    # In OpenCV cam: x=right, y=down, z=forward
    # up_corrected points upward in world → camera y = -up_corrected (y is down)
    R = np.array(
        [
            right,
            -up_corrected,  # camera y-axis points down
            forward,
        ]
    )  # (3, 3)

    # Translation: t = -R @ eye
    t = -R @ eye.reshape(3, 1)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t.ravel()
    return w2c


def intrinsic_matrix(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Compute 3×3 camera intrinsic matrix from horizontal FOV."""
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return K


def project_points(
    pts_world: np.ndarray,  # (N, 3)
    w2c: np.ndarray,         # (4, 4)
    K: np.ndarray,           # (3, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D pixel coordinates.

    Returns
    -------
    pts_2d : (N, 2) float pixel coordinates (x, y)
    depths  : (N,) float z-depths in camera space
    """
    N = pts_world.shape[0]
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts_world, ones]).T  # (4, N)
    cam = w2c @ pts_h  # (4, N)
    xyz_cam = cam[:3, :]  # (3, N)
    depths = xyz_cam[2, :]  # z-forward
    # Perspective divide
    valid = depths > 1e-4
    uv = np.full((2, N), np.nan)
    uv[:, valid] = (K[:2, :2] @ (xyz_cam[:2, valid] / xyz_cam[2:3, valid])) + K[:2, 2:3]
    return uv.T, depths  # (N, 2), (N,)


def world_to_cam(pts_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform world points to camera-space coordinates."""
    N = pts_world.shape[0]
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts_world, ones]).T
    cam = w2c @ pts_h
    return cam[:3, :].T  # (N, 3)


def compute_spatial_relations(
    centroid_a_world: np.ndarray,
    centroid_b_world: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    px_threshold: float = 20.0,
    depth_threshold: float = 0.1,
) -> dict:
    """Compute spatial relations between A and B as seen from the camera.

    Left/right and above/below use projected pixel coordinates so they match
    what is visually left/above in the rendered image (includes perspective
    division). Front/behind uses camera-space z (depth).

    Each axis has a dead zone: if the difference is smaller than the threshold,
    neither direction is true (both can be false, but both cannot be true).

    px_threshold    : pixel dead zone for left/right and above/below (default 20 px)
    depth_threshold : metre dead zone for front/behind (default 0.1 m)
    """
    pts = world_to_cam(np.stack([centroid_a_world, centroid_b_world]), w2c)
    a_cam, b_cam = pts[0], pts[1]
    uv, _ = project_points(np.stack([centroid_a_world, centroid_b_world]), w2c, K)
    a_px, b_px = uv[0], uv[1]

    dx = float(a_px[0] - b_px[0])
    dy = float(a_px[1] - b_px[1])
    dz = float(a_cam[2] - b_cam[2])

    return {
        "A_left_of_B":     dx < -px_threshold,
        "A_right_of_B":    dx >  px_threshold,
        "A_in_front_of_B": dz < -depth_threshold,
        "A_behind_B":      dz >  depth_threshold,
        "A_above_B":       dy < -px_threshold,   # y-down: smaller pixel-y is higher
        "A_below_B":       dy >  px_threshold,
    }


# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------

def compute_camera_candidates(
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    floor_y: float,
    camera_height: float,
    standoff_factor: float,
    standoff_min: float,
    standoff_max: float,
    up_idx: int = 1,
) -> list[dict]:
    """Analytically compute candidate camera positions that produce relation flips.

    Produces up to 4 positions:
      0 : left side of AB midpoint  (left/right flip pair with pos 1)
      1 : right side
      2 : AB-side (same as A, along AB axis) — front/behind flip pair with pos 3
      3 : BA-side (closer to B along axis)
      4 : diagonal

    Each dict has: eye (np.ndarray), target (np.ndarray), label (str).
    """
    M = (centroid_a + centroid_b) / 2.0

    # Ground-plane projection: zero out the up axis component
    d_ab_full = centroid_b - centroid_a
    d_ab_gnd = d_ab_full.copy()
    d_ab_gnd[up_idx] = 0.0
    dist_gnd = np.linalg.norm(d_ab_gnd)

    if dist_gnd < 1e-4:
        # Objects are vertically stacked — left/right flip is degenerate
        log.debug("Objects nearly vertically stacked; skipping pair")
        return []

    d_ab_gnd_norm = d_ab_gnd / dist_gnd

    # Perpendicular in the ground plane (rotate 90° around the up axis)
    # Works for both Y-up (idx=1) and Z-up (idx=2)
    axes = [0, 1, 2]
    ground_axes = [a for a in axes if a != up_idx]  # the two horizontal axes
    h0, h1 = ground_axes
    d_perp = np.zeros(3)
    d_perp[h0] = -d_ab_gnd_norm[h1]
    d_perp[h1] =  d_ab_gnd_norm[h0]

    # Standoff distance
    r = float(np.clip(standoff_factor * dist_gnd, standoff_min, standoff_max))

    # Eye height: floor + camera_height, clamped above the objects' midpoint height
    eye_h = floor_y + camera_height

    target = M.copy()
    target[up_idx] = (centroid_a[up_idx] + centroid_b[up_idx]) / 2.0

    candidates: list[dict] = []

    def make_eye(offset_3d: np.ndarray, label: str) -> dict:
        eye = M + offset_3d
        eye[up_idx] = eye_h
        return {"eye": eye.copy(), "target": target.copy(), "label": label}

    # Left/right flip positions (perpendicular to AB)
    candidates.append(make_eye(r * d_perp, "perp_pos"))
    candidates.append(make_eye(-r * d_perp, "perp_neg"))

    # Front/behind flip positions (along AB axis)
    candidates.append(make_eye(r * d_ab_gnd_norm, "along_pos"))
    candidates.append(make_eye(-r * d_ab_gnd_norm, "along_neg"))

    # Diagonal viewpoints
    diag1 = (d_ab_gnd_norm + d_perp)
    diag1 /= np.linalg.norm(diag1)
    diag2 = (d_ab_gnd_norm - d_perp)
    diag2 /= np.linalg.norm(diag2)
    candidates.append(make_eye(r * diag1, "diag_0"))
    candidates.append(make_eye(r * diag2, "diag_1"))

    return candidates


# ---------------------------------------------------------------------------
# Raycasting / validation
# ---------------------------------------------------------------------------

def build_raycasting_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    """Build an Open3D tensor raycasting scene from a legacy mesh."""
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    return scene


def nearest_geometry_distance(
    scene: o3d.t.geometry.RaycastingScene,
    point: np.ndarray,
) -> float:
    """Return distance from `point` to the nearest mesh surface."""
    query = o3d.core.Tensor(point.reshape(1, 3).astype(np.float32), dtype=o3d.core.Dtype.Float32)
    result = scene.compute_closest_points(query)
    closest = result["points"].numpy()  # (1, 3)
    return float(np.linalg.norm(closest[0] - point))


def check_occlusion(
    scene: o3d.t.geometry.RaycastingScene,
    camera_pos: np.ndarray,
    object_vertices: np.ndarray,  # (k, 3) world coords
    threshold: float,
    n_samples: int = 32,
) -> float:
    """Return fraction of sampled rays from camera to object vertices that are
    unobstructed (or reach within 5 cm of the target vertex).

    Returns fraction in [0, 1]. Higher = more visible.
    """
    if len(object_vertices) == 0:
        return 0.0

    # Sample up to n_samples vertices
    idx = np.random.choice(len(object_vertices), size=min(n_samples, len(object_vertices)), replace=False)
    targets = object_vertices[idx]

    origins = np.tile(camera_pos.astype(np.float32), (len(targets), 1))
    directions = targets.astype(np.float32) - origins
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    valid = (norms[:, 0] > 1e-4)
    directions[valid] = directions[valid] / norms[valid]
    expected_dist = norms[:, 0]

    rays = o3d.core.Tensor(
        np.hstack([origins, directions]).astype(np.float32),
        dtype=o3d.core.Dtype.Float32,
    )
    hits = scene.cast_rays(rays)
    hit_dist = hits["t_hit"].numpy()  # (N,)

    # A ray reaches the object if it hits at roughly the expected distance (±0.1m)
    reached = np.sum(
        (hit_dist >= expected_dist - 0.15) & (hit_dist < expected_dist + 0.15)
    )
    return float(reached) / len(targets)


def objects_in_frustum(
    centroid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    min_size: int,
) -> bool:
    """Check that the object's 2D projected bbox is at least min_size×min_size
    pixels and substantially within the image bounds."""
    # Sample 8 bbox corners + centroid
    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            centroid,
        ],
        dtype=np.float64,
    )
    pts_2d, depths = project_points(corners, w2c, K)
    front_mask = depths > 0.1
    if not np.any(front_mask):
        return False

    visible_pts = pts_2d[front_mask]
    in_frame = (
        (visible_pts[:, 0] >= 0)
        & (visible_pts[:, 0] < width)
        & (visible_pts[:, 1] >= 0)
        & (visible_pts[:, 1] < height)
    )
    if not np.any(in_frame):
        return False

    x_min = np.min(visible_pts[in_frame, 0])
    x_max = np.max(visible_pts[in_frame, 0])
    y_min = np.min(visible_pts[in_frame, 1])
    y_max = np.max(visible_pts[in_frame, 1])

    return (x_max - x_min) >= min_size and (y_max - y_min) >= min_size


def validate_camera(
    eye: np.ndarray,
    target: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    scene_rc: o3d.t.geometry.RaycastingScene,
    inst_a: dict,
    inst_b: dict,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    min_proj_size: int,
    occlusion_thresh: float,
    near_geom_dist: float,
    vertices: np.ndarray,
) -> bool:
    """Return True if this camera placement passes all validity checks."""
    # 1) Camera not too close to geometry
    d = nearest_geometry_distance(scene_rc, eye)
    if d < near_geom_dist:
        log.debug("Camera too close to geometry (%.3fm)", d)
        return False

    # 2) Occlusion check for both objects
    verts_a = vertices[inst_a["vertex_indices"]]
    vis_a = check_occlusion(scene_rc, eye, verts_a, occlusion_thresh)
    if vis_a < occlusion_thresh:
        log.debug("Object A occluded (%.1f%% visible)", vis_a * 100)
        return False

    verts_b = vertices[inst_b["vertex_indices"]]
    vis_b = check_occlusion(scene_rc, eye, verts_b, occlusion_thresh)
    if vis_b < occlusion_thresh:
        log.debug("Object B occluded (%.1f%% visible)", vis_b * 100)
        return False

    # 3) Both objects in frustum and large enough
    if not objects_in_frustum(
        inst_a["centroid"], inst_a["bbox_min"], inst_a["bbox_max"],
        w2c, K, width, height, min_proj_size,
    ):
        log.debug("Object A not sufficiently in frustum")
        return False
    if not objects_in_frustum(
        inst_b["centroid"], inst_b["bbox_min"], inst_b["bbox_max"],
        w2c, K, width, height, min_proj_size,
    ):
        log.debug("Object B not sufficiently in frustum")
        return False

    return True


def try_adjust_camera(
    eye_orig: np.ndarray,
    target: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    scene_rc: o3d.t.geometry.RaycastingScene,
    inst_a: dict,
    inst_b: dict,
    K: np.ndarray,
    width: int,
    height: int,
    min_proj_size: int,
    occlusion_thresh: float,
    near_geom_dist: float,
    vertices: np.ndarray,
    floor_y: float,
    up_idx: int = 1,
    n_steps: int = 5,
    step_size: float = 0.3,
    min_height_above_floor: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Try to find a valid camera position by stepping outward from eye_orig.

    The camera's up-axis coordinate is clamped to at least
    floor_y + min_height_above_floor at every step so the camera never dips
    below the floor surface, regardless of the stepping direction.

    Returns (eye, w2c) if found, else (None, None).
    """
    direction = eye_orig - target
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-6:
        return None, None
    direction = direction / d_norm

    floor_min = floor_y + min_height_above_floor

    for i in range(n_steps + 1):
        eye = eye_orig + direction * (i * step_size)
        if eye[up_idx] < floor_min:
            eye[up_idx] = floor_min
        world_up = np.zeros(3); world_up[up_idx] = 1.0
        w2c = look_at_matrix(eye, target, world_up)
        if validate_camera(
            eye, target, mesh, scene_rc, inst_a, inst_b,
            w2c, K, width, height, min_proj_size,
            occlusion_thresh, near_geom_dist, vertices,
        ):
            return eye, w2c

    return None, None


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _build_highlighted_colors(
    mesh: o3d.geometry.TriangleMesh,
    inst_a: dict,
    inst_b: dict,
    color_a_rgb: np.ndarray,
    color_b_rgb: np.ndarray,
) -> np.ndarray:
    """Return (N, 3) float64 vertex colour array where:

    - Object A vertices → color_a_rgb (float RGB in [0, 1])
    - Object B vertices → color_b_rgb
    - All other vertices → perceptual grayscale of their original colour
    """
    raw = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.ones((len(np.asarray(mesh.vertices)), 3), dtype=np.float64)
    )
    # Perceptual grayscale (ITU-R BT.601 luma)
    luma = 0.299 * raw[:, 0] + 0.587 * raw[:, 1] + 0.114 * raw[:, 2]
    result = np.stack([luma, luma, luma], axis=1).copy()
    result[inst_a["vertex_indices"]] = color_a_rgb
    result[inst_b["vertex_indices"]] = color_b_rgb
    return result


# ---------------------------------------------------------------------------
# Rendering  (uses PyVista/VTK for Windows-compatible headless rendering)
# Install: pip install pyvista
# ---------------------------------------------------------------------------

def _w2c_to_pyvista_camera(w2c: np.ndarray, K: np.ndarray, height: int):
    """Convert OpenCV w2c matrix + intrinsics to PyVista camera parameters.

    Returns (position, focal_point, up, view_angle_deg).
    - position    : camera eye in world coordinates
    - focal_point : a point along the look direction (1 m in front)
    - up          : world-space up vector (OpenCV y-down → negate cam y-axis)
    - view_angle  : vertical FOV in degrees
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    position = (-R.T @ t)                   # world position of camera
    forward_world = R[2, :]                 # camera z-axis in world space
    up_world = -R[1, :]                     # camera -y-axis = world up (y-down conv)
    focal_point = position + forward_world  # 1 m along look direction

    fy = K[1, 1]
    view_angle_deg = float(2.0 * math.degrees(math.atan(height / (2.0 * fy))))

    return position, focal_point, up_world, view_angle_deg


def _o3d_mesh_to_pyvista(mesh: o3d.geometry.TriangleMesh, override_colors: np.ndarray | None = None):
    """Convert an Open3D TriangleMesh to a PyVista PolyData with RGB point data.

    Parameters
    ----------
    override_colors : (N, 3) float64 in [0, 1], optional.
        If given, used instead of the mesh's own vertex_colors.
    """
    import pyvista as pv

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if override_colors is not None:
        raw_colors = override_colors
    elif mesh.has_vertex_colors():
        raw_colors = np.asarray(mesh.vertex_colors)
    else:
        raw_colors = np.ones((len(vertices), 3))

    colors_u8 = (np.clip(raw_colors, 0.0, 1.0) * 255).astype(np.uint8)

    # PyVista faces format: [n_pts, i0, i1, i2, ...]
    faces = np.hstack(
        [np.full((len(triangles), 1), 3, dtype=np.int32), triangles]
    ).ravel()

    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.point_data["RGB"] = colors_u8
    return pv_mesh


def _pyvista_render(
    pv_mesh,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    sphere_center: np.ndarray | None = None,
    sphere_color: np.ndarray | None = None,
    arrow_target: np.ndarray | None = None,
) -> np.ndarray:
    """Render a PyVista mesh with the given camera and return (H, W, 3) uint8.

    If sphere_center and arrow_target are provided, an arrow is drawn at sphere_center
    pointing towards arrow_target (the midpoint between the two highlighted object centroids),
    coloured with sphere_color (float RGB in [0, 1]).
    """
    import pyvista as pv

    position, focal_point, up_world, view_angle = _w2c_to_pyvista_camera(w2c, K, height)

    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.set_background("black")
    plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, show_scalar_bar=False)

    if sphere_center is not None and arrow_target is not None:
        direction = arrow_target - sphere_center
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            direction_unit = direction / length
            arrow = pv.Arrow(
                start=sphere_center.tolist(),
                direction=direction_unit.tolist(),
                scale=min(length, 0.5),
                tip_length=0.25,
                tip_radius=0.1,
                shaft_radius=0.04,
            )
            plotter.add_mesh(arrow, color=np.clip(sphere_color, 0.0, 1.0).tolist())

    plotter.camera.position = position.tolist()
    plotter.camera.focal_point = focal_point.tolist()
    plotter.camera.up = up_world.tolist()
    plotter.camera.view_angle = view_angle

    img = plotter.screenshot(return_img=True, window_size=[width, height])
    plotter.close()

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def render_scene(
    mesh: o3d.geometry.TriangleMesh,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    inst_a: dict | None = None,
    inst_b: dict | None = None,
    color_a_rgb: np.ndarray | None = None,
    color_b_rgb: np.ndarray | None = None,
    sphere_center: np.ndarray | None = None,
    sphere_color: np.ndarray | None = None,
    arrow_target: np.ndarray | None = None,
) -> np.ndarray:
    """Render the scene using PyVista (VTK-based, Windows headless compatible).

    When inst_a/inst_b and their colours are supplied, object A and B are
    rendered in their assigned highlight colours; everything else is converted
    to perceptual grayscale.

    When sphere_center and arrow_target are provided, an arrow is rendered at
    sphere_center pointing towards arrow_target (midpoint between object centroids),
    using sphere_color (float RGB in [0, 1]).

    Returns (H, W, 3) uint8 RGB array.
    """
    if inst_a is not None and inst_b is not None and color_a_rgb is not None:
        override = _build_highlighted_colors(mesh, inst_a, inst_b, color_a_rgb, color_b_rgb)
        pv_mesh = _o3d_mesh_to_pyvista(mesh, override_colors=override)
    else:
        pv_mesh = _o3d_mesh_to_pyvista(mesh)
    return _pyvista_render(pv_mesh, w2c, K, width, height, sphere_center=sphere_center, sphere_color=sphere_color, arrow_target=arrow_target)


def render_instance_mask(
    mesh: o3d.geometry.TriangleMesh,
    inst_a: dict,
    inst_b: dict,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[list | None, list | None]:
    """Render an instance-colour mask and derive 2D bounding boxes.

    Assigns red to object A, green to object B, black to everything else.
    Returns (bbox_a, bbox_b) where each bbox is [x1, y1, x2, y2] or None.
    """
    n_verts = len(np.asarray(mesh.vertices))
    colors = np.zeros((n_verts, 3), dtype=np.float64)
    colors[inst_a["vertex_indices"]] = [1.0, 0.0, 0.0]
    colors[inst_b["vertex_indices"]] = [0.0, 1.0, 0.0]

    pv_mesh = _o3d_mesh_to_pyvista(mesh, override_colors=colors)
    img = _pyvista_render(pv_mesh, w2c, K, width, height)

    def bbox_from_mask(channel_idx: int, threshold: int = 100) -> list[int] | None:
        mask = img[:, :, channel_idx] > threshold
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return bbox_from_mask(0), bbox_from_mask(1)  # red=A, green=B


# ---------------------------------------------------------------------------
# Pair selection and angular separation
# ---------------------------------------------------------------------------

def angular_separation(eye0: np.ndarray, eye1: np.ndarray, focus: np.ndarray) -> float:
    """Return angle in degrees between two camera rays from the focus point."""
    v0 = eye0 - focus
    v1 = eye1 - focus
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v0 / n0, v1 / n1), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_a)))


def detect_up_axis(axis_mat: np.ndarray, vertices: np.ndarray) -> int:
    """Detect which world axis (0=X,1=Y,2=Z) is 'up' after alignment.

    Strategy: the vertical axis (floor→ceiling) has the smallest coordinate
    range in the aligned mesh, since rooms are typically wider than they are
    tall.  Ties are broken by which row of the rotation matrix is most aligned
    with a single standard-basis vector (i.e. is most "pure"), which catches
    scenes where the room footprint is square.
    """
    ranges = vertices.max(axis=0) - vertices.min(axis=0)
    # Primary: smallest range = up axis
    # Secondary (tie-break): row of R that is closest to a unit basis vector
    R = axis_mat[:3, :3]
    row_purity = np.array([np.max(np.abs(R[i, :])) for i in range(3)])
    # Combine: invert range (small range → large score), add purity as tie-break
    score = -ranges + 0.01 * row_purity
    return int(np.argmax(score))


# ---------------------------------------------------------------------------
# Arrow placement
# ---------------------------------------------------------------------------

def _point_inside_any_instance(
    pos: np.ndarray,
    instances: list[dict],
    margin: float = 0.05,
) -> bool:
    """Return True if pos falls inside the axis-aligned bounding box of any instance.

    A small margin is added to each bbox to catch points that are on or just
    outside a surface but would still cause the arrow to appear embedded.
    """
    for inst in instances:
        lo = inst["bbox_min"] - margin
        hi = inst["bbox_max"] + margin
        if np.all(pos >= lo) and np.all(pos <= hi):
            return True
    return False


def _arrow_visible_from_viewpoint(
    scene_rc: o3d.t.geometry.RaycastingScene,
    vp: dict,
    arrow_pos: np.ndarray,
    arrow_target: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    occlusion_thresh: float,
    arrow_scale: float = 0.5,
) -> bool:
    """Return True if the arrow is visible from the given viewpoint.

    Samples three points along the arrow shaft (base, mid, near-tip) and
    requires that all are in front of the camera, that the fraction of
    unoccluded samples meets occlusion_thresh, and that at least one sample
    projects inside the image frame.
    """
    eye     = vp["eye"]
    w2c     = vp["w2c"]
    forward = w2c[2, :3]  # camera z-axis in world space

    direction = arrow_target - arrow_pos
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return False
    direction_unit = direction / length
    scale = min(length, arrow_scale)

    # Three sample points along the arrow
    samples = np.array([
        arrow_pos,
        arrow_pos + direction_unit * scale * 0.5,
        arrow_pos + direction_unit * scale * 0.85,
    ])

    # 1. All samples must be in front of the camera
    for pt in samples:
        if float(np.dot(pt - eye, forward)) <= 0.0:
            return False

    # 2. Occlusion check — same logic as for highlighted objects
    vis = check_occlusion(scene_rc, eye, samples, occlusion_thresh, n_samples=len(samples))
    if vis < occlusion_thresh:
        return False

    # 3. At least one sample must project within the image frame
    pts_2d, depths = project_points(samples, w2c, K)
    return any(
        d > 0.1 and 0 <= u < width and 0 <= v < height
        for (u, v), d in zip(pts_2d, depths)
    )


def find_arrow_position(
    scene_rc: o3d.t.geometry.RaycastingScene,
    viewpoints: list[dict],
    rels: list[dict],
    ca: np.ndarray,
    cb: np.ndarray,
    inst_a: dict,
    inst_b: dict,
    instances: list[dict],
    floor_y: float,
    up_idx: int,
    vertices: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    occlusion_thresh: float,
    min_proj_size: int,
    arrow_radius: float = 0.08,
    arrow_height_above_floor: float = 0.7,
    n_random: int = 300,
    min_visible: int = 2,
) -> tuple[np.ndarray | None, list[dict], list[dict]]:
    """Find a world position for the arrow visible from at least min_visible viewpoints.

    Two extra placement constraints beyond raycasting occlusion:

    1. Minimum distance from the centroid midpoint — the arrow must be at least
       max(longest_bbox_dim_of_A_or_B, centroid_distance) away from the midpoint
       (measured in the ground plane), so it is never visually embedded in or
       immediately adjacent to either highlighted object.

    2. Not inside any instance bounding box — checked via
       _point_inside_any_instance to prevent the arrow from appearing embedded
       inside furniture or other scene objects.

    Candidate positions are generated from structured heuristics first (offsets
    along/perpendicular to the A-B axis and diagonals, including offsets at
    exactly min_arrow_dist), then random sampling within the scene bounds as a
    fallback.  Visibility is tested with the same raycasting occlusion logic
    used for the highlighted objects.

    Returns (arrow_pos, visible_viewpoints, visible_rels) — or (None, [], [])
    if no suitable position is found.
    """
    arrow_target = (ca + cb) / 2.0
    arrow_h = floor_y + arrow_height_above_floor

    ground_axes = [a for a in (0, 1, 2) if a != up_idx]
    h0, h1 = ground_axes

    midpoint = (ca + cb) / 2.0

    # Minimum ground-plane distance from midpoint:
    # max(longest bbox dimension of either object, centroid-to-centroid distance).
    len_a = float(np.max(inst_a["bbox_max"] - inst_a["bbox_min"]))
    len_b = float(np.max(inst_b["bbox_max"] - inst_b["bbox_min"]))
    centroid_dist = float(np.linalg.norm(ca - cb))
    min_arrow_dist = max(len_a, len_b, centroid_dist)

    d_ab = (cb - ca).copy()
    d_ab[up_idx] = 0.0
    d_ab_len = float(np.linalg.norm(d_ab))

    candidates: list[np.ndarray] = []

    if d_ab_len > 1e-6:
        d_ab_norm = d_ab / d_ab_len
        d_perp = np.zeros(3)
        d_perp[h0] = -d_ab_norm[h1]
        d_perp[h1] =  d_ab_norm[h0]

        # Perpendicular offsets from midpoint — start at min_arrow_dist
        for dist in (min_arrow_dist, min_arrow_dist * 1.25, min_arrow_dist * 1.5,
                     min_arrow_dist * 1.75, min_arrow_dist * 2.0):
            for sign in (1, -1):
                pt = midpoint.copy()
                pt[up_idx] = arrow_h
                pt += sign * dist * d_perp
                candidates.append(pt.copy())

        # Along-axis offsets beyond midpoint — start at min_arrow_dist
        for dist in (min_arrow_dist, min_arrow_dist * 1.25, min_arrow_dist * 1.5):
            for sign in (1, -1):
                pt = midpoint.copy()
                pt[up_idx] = arrow_h
                pt += sign * dist * d_ab_norm
                candidates.append(pt.copy())

        # Diagonals in all four quadrants at min_arrow_dist and beyond
        for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            diag = s1 * d_ab_norm + s2 * d_perp
            diag = diag / np.linalg.norm(diag)
            for dist in (min_arrow_dist, min_arrow_dist * 1.25, min_arrow_dist * 1.5):
                pt = midpoint.copy()
                pt[up_idx] = arrow_h
                pt += dist * diag
                candidates.append(pt.copy())
    else:
        pt = midpoint.copy()
        pt[up_idx] = arrow_h
        candidates.append(pt.copy())

    # Random fallback within the 10th–90th-percentile scene bounding box
    lo0 = float(np.percentile(vertices[:, h0], 10))
    hi0 = float(np.percentile(vertices[:, h0], 90))
    lo1 = float(np.percentile(vertices[:, h1], 10))
    hi1 = float(np.percentile(vertices[:, h1], 90))
    for _ in range(n_random):
        pt = np.zeros(3)
        pt[h0]     = np.random.uniform(lo0, hi0)
        pt[h1]     = np.random.uniform(lo1, hi1)
        pt[up_idx] = arrow_h
        candidates.append(pt)

    # Evaluate each candidate
    for pos in candidates:
        # Ground-plane distance from midpoint must meet minimum
        gnd_dist = float(np.linalg.norm((pos - midpoint)[[h0, h1]]))
        if gnd_dist < min_arrow_dist:
            continue

        # Must not clip into scene mesh geometry
        if nearest_geometry_distance(scene_rc, pos) < arrow_radius:
            continue

        # Must not be inside any instance's bounding box
        if _point_inside_any_instance(pos, instances):
            continue

        # Both highlighted objects must be sufficiently visible from the arrow's
        # own pose (same occlusion + frustum checks as validate_camera).
        world_up_vec = np.zeros(3); world_up_vec[up_idx] = 1.0
        w2c_cand = look_at_matrix(pos, arrow_target, world_up_vec)
        if check_occlusion(scene_rc, pos, vertices[inst_a["vertex_indices"]], occlusion_thresh) < occlusion_thresh:
            continue
        if check_occlusion(scene_rc, pos, vertices[inst_b["vertex_indices"]], occlusion_thresh) < occlusion_thresh:
            continue
        if not objects_in_frustum(inst_a["centroid"], inst_a["bbox_min"], inst_a["bbox_max"], w2c_cand, K, width, height, min_proj_size):
            continue
        if not objects_in_frustum(inst_b["centroid"], inst_b["bbox_min"], inst_b["bbox_max"], w2c_cand, K, width, height, min_proj_size):
            continue

        visible_vps:  list[dict] = []
        visible_rels: list[dict] = []
        for vp, rel in zip(viewpoints, rels):
            if _arrow_visible_from_viewpoint(
                scene_rc, vp, pos, arrow_target,
                K, width, height, occlusion_thresh,
            ):
                visible_vps.append(vp)
                visible_rels.append(rel)

        if len(visible_vps) >= min_visible:
            return pos, visible_vps, visible_rels

    return None, [], []


# ---------------------------------------------------------------------------
# Main scene processing
# ---------------------------------------------------------------------------

def process_scene(
    scene_dir: Path,
    output_dir: Path,
    *,
    skip_labels: set[str],
    min_object_volume: float,
    min_centroid_dist: float,
    max_centroid_dist: float,
    standoff_factor: float,
    standoff_min: float,
    standoff_max: float,
    camera_height: float,
    fov: float,
    width: int,
    height: int,
    min_proj_size: int,
    occlusion_thresh: float,
    max_pairs: int,
    near_geom_dist: float,
    full_colour: bool = False,
    reference_object: bool = False,
    print_reference_image: bool = False,
) -> None:
    """Process a single ScanNet scene directory."""
    scene_id = scene_dir.name
    log.info("=== Processing scene: %s ===", scene_id)

    # Locate files
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    labels_ply_path = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    txt_path = scene_dir / f"{scene_id}.txt"

    for p in [ply_path, agg_path, segs_path]:
        if not p.exists():
            log.error("Required file missing: %s — skipping scene", p)
            return

    # Load axis alignment
    axis_mat = load_axis_alignment(txt_path)

    # Load mesh and instances (we need aligned vertices to detect up axis)
    mesh, instances = load_mesh_and_instances(
        ply_path, labels_ply_path, agg_path, segs_path,
        axis_mat, skip_labels, min_object_volume,
    )

    if len(instances) < 2:
        log.warning("Scene %s has fewer than 2 valid instances — skipping", scene_id)
        return

    vertices = np.asarray(mesh.vertices)

    up_axis = detect_up_axis(axis_mat, vertices)
    log.info("Up axis after alignment: %d (0=X,1=Y,2=Z)", up_axis)

    # Floor height: use the 2nd percentile of the up-axis coordinate rather than
    # the absolute minimum, which can be pulled down by stray vertices or mesh
    # artifacts that sit below the actual floor surface.
    up_idx = up_axis  # 0=X, 1=Y, 2=Z
    floor_y = float(np.percentile(vertices[:, up_idx], 2))
    log.info(
        "Floor height (2nd-percentile of axis %d): %.3f m  "
        "(abs min was %.3f m)",
        up_idx, floor_y, float(vertices[:, up_idx].min()),
    )
    if up_axis == 2:
        log.warning("Z-up scene detected; camera height logic uses Z axis")

    # Camera intrinsics
    K = intrinsic_matrix(fov, width, height)

    # Build raycasting scene once
    log.info("Building raycasting scene...")
    scene_rc = build_raycasting_scene(mesh)

    # Output dirs
    scene_out = output_dir / scene_id
    img_dir = scene_out / "images"
    vp_dir = output_dir / "viewpoints" / f"scene{scene_id.replace('scene', '')}"
    img_dir.mkdir(parents=True, exist_ok=True)
    vp_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate pairs
    all_pairs = list(combinations(range(len(instances)), 2))
    np.random.shuffle(all_pairs)

    viewpoint_groups: list[dict] = []
    pair_color_index = 0  # incremented only for accepted pairs

    for idx_a, idx_b in all_pairs:
        if len(viewpoint_groups) >= max_pairs:
            break

        inst_a = instances[idx_a]
        inst_b = instances[idx_b]

        ca = inst_a["centroid"]
        cb = inst_b["centroid"]
        dist = float(np.linalg.norm(ca - cb))

        if not (min_centroid_dist <= dist <= max_centroid_dist):
            continue

        pair_id = f"{inst_a['instance_id']}_{inst_b['instance_id']}"
        log.info(
            "Pair %s: %s ↔ %s (dist=%.2fm)",
            pair_id, inst_a["label"], inst_b["label"], dist,
        )

        # Assign highlight colours for this pair (A, B, and optional sphere)
        color_name_a, color_rgb_a, color_name_b, color_rgb_b, color_name_sp, color_rgb_sp = assign_pair_colors(pair_color_index)

        # Compute candidate camera positions
        candidates = compute_camera_candidates(
            ca, cb, floor_y, camera_height,
            standoff_factor, standoff_min, standoff_max,
            up_idx=up_axis,
        )
        if not candidates:
            continue

        # Validate each candidate
        valid_viewpoints: list[dict] = []
        for cand in candidates:
            eye_cand = cand["eye"]
            tgt_cand = cand["target"]
            world_up = np.zeros(3); world_up[up_axis] = 1.0

            w2c_cand = look_at_matrix(eye_cand, tgt_cand, world_up)

            eye_final, w2c_final = try_adjust_camera(
                eye_cand, tgt_cand, mesh, scene_rc,
                inst_a, inst_b, K, width, height,
                min_proj_size, occlusion_thresh, near_geom_dist, vertices,
                floor_y=floor_y,
                up_idx=up_axis,
            )
            if eye_final is None:
                log.debug("Candidate %s invalid after adjustment", cand["label"])
                continue

            valid_viewpoints.append(
                {
                    "eye": eye_final,
                    "target": tgt_cand,
                    "w2c": w2c_final,
                    "label": cand["label"],
                }
            )

        if len(valid_viewpoints) < 2:
            log.info("  Fewer than 2 valid viewpoints — skipping pair")
            continue

        # Compute spatial relations for all valid viewpoints
        rels = [
            compute_spatial_relations(ca, cb, vp["w2c"], K)
            for vp in valid_viewpoints
        ]

        # Check that at least one flip occurs (left/right or front/behind)
        flipped: list[str] = []
        r0 = rels[0]
        for ri in rels[1:]:
            if ri["A_left_of_B"] != r0["A_left_of_B"] or ri["A_right_of_B"] != r0["A_right_of_B"]:
                if "left_right" not in flipped:
                    flipped.append("left_right")
            if ri["A_in_front_of_B"] != r0["A_in_front_of_B"] or ri["A_behind_B"] != r0["A_behind_B"]:
                if "front_behind" not in flipped:
                    flipped.append("front_behind")

        if not flipped:
            log.info("  No relation flip detected — skipping pair")
            continue

        # Cap at 3 viewpoints (prefer the flip pair + one diagonal)
        # Ensure we keep viewpoints from both sides of the flip
        selected_vps: list[dict] = []
        selected_rels: list[dict] = []
        # Keep at most one from each "side" (perp_pos/perp_neg, along_pos/along_neg)
        seen_sides = set()
        for vp, rel in zip(valid_viewpoints, rels):
            if len(selected_vps) >= 3:
                break
            side_key = vp["label"]
            if side_key not in seen_sides:
                selected_vps.append(vp)
                selected_rels.append(rel)
                seen_sides.add(side_key)

        if len(selected_vps) < 2:
            continue

        # Find an arrow position visible from at least 2 of the selected viewpoints.
        # Viewpoint selection is done first (independent of the arrow); arrow
        # placement is solved afterwards with structured heuristics + random
        # sampling, using the same occlusion raycasting as for the objects.
        sphere_pos: np.ndarray | None = None
        if reference_object:
            sphere_pos, arrow_vps, arrow_rels = find_arrow_position(
                scene_rc, selected_vps, selected_rels, ca, cb,
                inst_a, inst_b, instances,
                floor_y, up_axis, vertices, K, width, height, occlusion_thresh, min_proj_size,
            )
            if sphere_pos is None:
                log.info(
                    "  Pair %s: no arrow position visible from 2+ viewpoints — skipping pair",
                    pair_id,
                )
                continue
            selected_vps  = arrow_vps
            selected_rels = arrow_rels

            # Re-check that a relation flip still holds among the viewpoints that
            # can see the arrow (filtering may have dropped one side of the flip).
            flipped = []
            r0 = selected_rels[0]
            for ri in selected_rels[1:]:
                if ri["A_left_of_B"] != r0["A_left_of_B"] or ri["A_right_of_B"] != r0["A_right_of_B"]:
                    if "left_right" not in flipped:
                        flipped.append("left_right")
                if ri["A_in_front_of_B"] != r0["A_in_front_of_B"] or ri["A_behind_B"] != r0["A_behind_B"]:
                    if "front_behind" not in flipped:
                        flipped.append("front_behind")
            if not flipped:
                log.info("  Pair %s: arrow-visibility filtering removed the flip — skipping pair", pair_id)
                continue

        # Arrow-perspective spatial relations and optional reference image
        arrow_target_pos: np.ndarray | None = None
        arrow_spatial_rels: dict | None = None
        arrow_image_path: str | None = None
        arrow_pose: dict | None = None
        if sphere_pos is not None:
            arrow_target_pos = (ca + cb) / 2.0
            world_up = np.zeros(3); world_up[up_axis] = 1.0
            w2c_arrow = look_at_matrix(sphere_pos, arrow_target_pos, world_up)
            arrow_spatial_rels = compute_spatial_relations(ca, cb, w2c_arrow, K)

            # Pose description: position, forward/right/up axes, and the world-up
            # convention used (camera up = world up = floor normal).
            forward_vec = (arrow_target_pos - sphere_pos)
            forward_vec = forward_vec / np.linalg.norm(forward_vec)
            right_vec   = np.cross(forward_vec, world_up)
            right_norm  = np.linalg.norm(right_vec)
            if right_norm > 1e-6:
                right_vec = right_vec / right_norm
            else:
                # Degenerate case: arrow points straight up/down — fall back to x-axis
                right_vec = np.array([1.0, 0.0, 0.0])
                right_vec[up_axis] = 0.0
                right_vec = right_vec / np.linalg.norm(right_vec)
            up_vec = np.cross(right_vec, forward_vec)

            arrow_pose = {
                "position_world":  [round(float(v), 4) for v in sphere_pos],
                "forward_world":   [round(float(v), 4) for v in forward_vec],
                "right_world":     [round(float(v), 4) for v in right_vec],
                "up_world":        [round(float(v), 4) for v in up_vec],
                "up_convention":   "world_up — camera up-axis is aligned with the floor normal (axis %d)" % up_axis,
                "w2c_matrix":      [[round(float(v), 6) for v in row] for row in w2c_arrow.tolist()],
                "fov_degrees":     fov,
                "image_resolution": [width, height],
            }

            if print_reference_image:
                arrow_img_name = f"objA_{inst_a['instance_id']}_objB_{inst_b['instance_id']}_view_arrow.png"
                arrow_img_path = img_dir / arrow_img_name
                arrow_vp_path  = vp_dir  / arrow_img_name
                try:
                    rgb_arrow = render_scene(
                        mesh, w2c_arrow, K, width, height,
                        inst_a=None if full_colour else inst_a,
                        inst_b=None if full_colour else inst_b,
                        color_a_rgb=None if full_colour else color_rgb_a,
                        color_b_rgb=None if full_colour else color_rgb_b,
                    )
                    Image.fromarray(rgb_arrow).save(str(arrow_img_path))
                    Image.fromarray(rgb_arrow).save(str(arrow_vp_path))
                    arrow_image_path = f"images/{arrow_img_name}"
                except Exception as exc:
                    log.warning("  Arrow-view render failed: %s", exc)

        # Render images and compute 2D bboxes
        viewpoint_records: list[dict] = []
        any_render_failed = False

        for vi, (vp, rel) in enumerate(zip(selected_vps, selected_rels)):
            img_name = f"objA_{inst_a['instance_id']}_objB_{inst_b['instance_id']}_view_{vi}.png"
            img_path = img_dir / img_name
            vp_img_path = vp_dir / img_name

            try:
                rgb = render_scene(
                    mesh, vp["w2c"], K, width, height,
                    inst_a=None if full_colour else inst_a,
                    inst_b=None if full_colour else inst_b,
                    color_a_rgb=None if full_colour else color_rgb_a,
                    color_b_rgb=None if full_colour else color_rgb_b,
                    sphere_center=sphere_pos,
                    sphere_color=color_rgb_sp if sphere_pos is not None else None,
                    arrow_target=(ca + cb) / 2.0 if sphere_pos is not None else None,
                )
                Image.fromarray(rgb).save(str(img_path))
                Image.fromarray(rgb).save(str(vp_img_path))
            except Exception as exc:
                log.warning("  Render failed for viewpoint %d: %s", vi, exc)
                any_render_failed = True
                break

            # 2D bounding boxes via instance mask render
            try:
                bbox_a, bbox_b = render_instance_mask(
                    mesh, inst_a, inst_b, vp["w2c"], K, width, height
                )
            except Exception as exc:
                log.warning("  Instance mask failed: %s — falling back to projected bbox", exc)
                bbox_a = None
                bbox_b = None

            # Fallback: project bbox corners
            if bbox_a is None:
                pts_2d_a, depths_a = project_points(
                    np.array([inst_a["centroid"]]), vp["w2c"], K
                )
                if depths_a[0] > 0:
                    u, v_ = float(pts_2d_a[0, 0]), float(pts_2d_a[0, 1])
                    bbox_a = [max(0, int(u) - 25), max(0, int(v_) - 25),
                              min(width, int(u) + 25), min(height, int(v_) + 25)]
                else:
                    bbox_a = [0, 0, 0, 0]

            if bbox_b is None:
                pts_2d_b, depths_b = project_points(
                    np.array([inst_b["centroid"]]), vp["w2c"], K
                )
                if depths_b[0] > 0:
                    u, v_ = float(pts_2d_b[0, 0]), float(pts_2d_b[0, 1])
                    bbox_b = [max(0, int(u) - 25), max(0, int(v_) - 25),
                              min(width, int(u) + 25), min(height, int(v_) + 25)]
                else:
                    bbox_b = [0, 0, 0, 0]

            # Angular separation with previous viewpoint
            ang_sep = 0.0
            if vi > 0:
                ang_sep = angular_separation(
                    selected_vps[0]["eye"], vp["eye"],
                    (ca + cb) / 2.0,
                )

            viewpoint_records.append(
                {
                    "viewpoint_index": vi,
                    "image_path": f"images/{img_name}",
                    "fov_degrees": fov,
                    "image_resolution": [width, height],
                    "spatial_relations": rel,
                    "viewpoint_label": vp["label"],
                    "angular_sep_from_view0_deg": round(ang_sep, 2),
                }
            )

        if any_render_failed or len(viewpoint_records) < 2:
            continue

        # Overall angular separation (view 0 → view 1)
        overall_ang = angular_separation(
            selected_vps[0]["eye"], selected_vps[1]["eye"],
            (ca + cb) / 2.0,
        )

        group = {
            "pair_id": pair_id,
            "object_A": {
                "instance_id": inst_a["instance_id"],
                "label": inst_a["label"],
                "color": color_name_a,
                "color_rgb": [round(float(v), 3) for v in color_rgb_a],
            },
            "object_B": {
                "instance_id": inst_b["instance_id"],
                "label": inst_b["label"],
                "color": color_name_b,
                "color_rgb": [round(float(v), 3) for v in color_rgb_b],
            },
            **({"reference_object_arrow": {
                "color": color_name_sp,
                "color_rgb": [round(float(v), 3) for v in color_rgb_sp],
                "pose": arrow_pose,
                "spatial_relations_from_arrow": arrow_spatial_rels,
                **({"image_path": arrow_image_path} if arrow_image_path is not None else {}),
            }} if reference_object else {}),
            "viewpoints": viewpoint_records,
            "flipped_relations": flipped,
            "viewpoint_angular_separation_degrees": round(overall_ang, 2),
        }
        viewpoint_groups.append(group)
        pair_color_index += 1
        log.info(
            "  -> Saved pair %s: %d viewpoints, flips=%s, ang_sep=%.1f°, colors=%s/%s",
            pair_id, len(viewpoint_records), flipped, overall_ang, color_name_a, color_name_b,
        )

    # Save metadata JSON
    metadata = {
        "scene_id": scene_id,
        "axis_alignment_applied": True,
        "up_axis": ["X", "Y", "Z"][up_axis],
        "camera_conventions": "OpenCV (x-right, y-down, z-forward)",
        "viewpoint_groups": viewpoint_groups,
    }
    meta_path = scene_out / "metadata.json"
    with meta_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)

    log.info(
        "Scene %s done: %d viewpoint groups saved to %s",
        scene_id, len(viewpoint_groups), scene_out,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate viewpoint pairs from ScanNet scenes for VLM spatial reasoning evaluation."
    )
    p.add_argument(
        "--scene_dir",
        required=True,
        help="Path to a single scene dir OR to the root scannet dir (use with --batch).",
    )
    p.add_argument(
        "--output_dir",
        default="outputs",
        help="Root output directory. Default: outputs/",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Treat --scene_dir as a root containing multiple scene subdirectories.",
    )
    p.add_argument("--min_object_volume", type=float, default=DEFAULT_MIN_OBJECT_VOLUME)
    p.add_argument("--min_centroid_distance", type=float, default=DEFAULT_MIN_CENTROID_DIST)
    p.add_argument("--max_centroid_distance", type=float, default=DEFAULT_MAX_CENTROID_DIST)
    p.add_argument("--standoff_distance_factor", type=float, default=DEFAULT_STANDOFF_FACTOR)
    p.add_argument("--standoff_min", type=float, default=DEFAULT_STANDOFF_MIN)
    p.add_argument("--standoff_max", type=float, default=DEFAULT_STANDOFF_MAX)
    p.add_argument("--camera_height", type=float, default=DEFAULT_CAMERA_HEIGHT)
    p.add_argument("--fov", type=float, default=DEFAULT_FOV)
    p.add_argument("--resolution_w", type=int, default=DEFAULT_RES_W)
    p.add_argument("--resolution_h", type=int, default=DEFAULT_RES_H)
    p.add_argument("--min_projected_size", type=int, default=DEFAULT_MIN_PROJ_SIZE)
    p.add_argument("--occlusion_ray_threshold", type=float, default=DEFAULT_OCCLUSION_THRESH)
    p.add_argument("--max_pairs_per_scene", type=int, default=DEFAULT_MAX_PAIRS)
    p.add_argument(
        "--skip_labels",
        nargs="+",
        default=None,
        help="Space-separated list of object labels to skip. Overrides default list.",
    )
    p.add_argument("--near_geom_dist", type=float, default=DEFAULT_NEAR_GEOM_DIST)
    p.add_argument(
        "--full-colour",
        action="store_true",
        default=False,
        help="Render in original scene colours without highlighting objects or converting to grayscale.",
    )
    p.add_argument(
        "--reference-object",
        action="store_true",
        default=False,
        help="Add a coloured arrow pointing toward the midpoint between the two highlighted objects, placed at a position visible from at least 2 viewpoints.",
    )
    p.add_argument(
        "--print-reference-image",
        action="store_true",
        default=False,
        help="Render an additional image from the arrow's viewpoint and save it as objA_x_objB_y_view_arrow.png. Requires --reference-object.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    np.random.seed(args.seed)

    skip_labels = set(args.skip_labels) if args.skip_labels is not None else DEFAULT_SKIP_LABELS

    kwargs = dict(
        skip_labels=skip_labels,
        min_object_volume=args.min_object_volume,
        min_centroid_dist=args.min_centroid_distance,
        max_centroid_dist=args.max_centroid_distance,
        standoff_factor=args.standoff_distance_factor,
        standoff_min=args.standoff_min,
        standoff_max=args.standoff_max,
        camera_height=args.camera_height,
        fov=args.fov,
        width=args.resolution_w,
        height=args.resolution_h,
        min_proj_size=args.min_projected_size,
        occlusion_thresh=args.occlusion_ray_threshold,
        max_pairs=args.max_pairs_per_scene,
        near_geom_dist=args.near_geom_dist,
        full_colour=args.full_colour,
        reference_object=args.reference_object,
        print_reference_image=args.print_reference_image,
    )

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        scene_dirs = sorted(
            d for d in scene_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene")
        )
        log.info("Batch mode: found %d scene directories", len(scene_dirs))
        for sd in scene_dirs:
            try:
                process_scene(sd, output_dir, **kwargs)
            except Exception as exc:
                log.exception("Scene %s failed: %s", sd.name, exc)
    else:
        process_scene(scene_dir, output_dir, **kwargs)


if __name__ == "__main__":
    main()
