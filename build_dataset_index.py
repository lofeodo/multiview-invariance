"""
build_dataset_index.py

Scans all per-scene output directories and builds a consolidated dataset index.

The raw per-scene outputs (outputs/<scene_id>/metadata.json + images/) are
left untouched.  This script only reads them and writes three JSONL files to
a separate index directory:

    dataset/
        scenes.jsonl    — one record per scene
        groups.jsonl    — one record per object pair (group)
        examples.jsonl  — one record per rendered viewpoint (example)

IDs follow the convention:
    group_id   = {scene_id}__{pair_id}
    example_id = {scene_id}__{pair_id}__view_{viewpoint_index}

Image paths in examples.jsonl are relative to the project root, e.g.
    outputs/scene0700_00/images/objA_10_objB_11_view_0.png

Usage:
    python build_dataset_index.py
    python build_dataset_index.py --outputs_dir outputs --index_dir dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index(outputs_dir: Path, index_dir: Path) -> None:
    """Scan *outputs_dir* and write the consolidated index to *index_dir*.

    Parameters
    ----------
    outputs_dir:
        Root directory that contains one sub-folder per scene, each with a
        ``metadata.json`` file and an ``images/`` sub-directory.
    index_dir:
        Destination directory for the three JSONL index files.  Created if it
        does not yet exist.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[dict] = []
    groups: list[dict] = []
    examples: list[dict] = []

    # Discover all per-scene metadata files, sorted for reproducibility.
    metadata_files = sorted(outputs_dir.glob("*/metadata.json"))

    if not metadata_files:
        print(f"No metadata.json files found under {outputs_dir}/")
        return

    print(f"Found {len(metadata_files)} scene(s). Building index...")

    for meta_path in metadata_files:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        scene_id: str = meta["scene_id"]
        viewpoint_groups: list[dict] = meta.get("viewpoint_groups", [])

        # --- Scene record ---
        scenes.append(
            {
                "scene_id": scene_id,
                "n_groups": len(viewpoint_groups),
            }
        )

        for vg in viewpoint_groups:
            pair_id: str = vg["pair_id"]
            group_id = f"{scene_id}__{pair_id}"
            viewpoints: list[dict] = vg.get("viewpoints", [])

            # --- Group record ---
            group_record: dict = {
                "group_id": group_id,
                "scene_id": scene_id,
                "pair_id": pair_id,
                "object_A": vg["object_A"],
                "object_B": vg["object_B"],
                "n_viewpoints": len(viewpoints),
                "viewpoint_angular_separation_degrees": vg.get(
                    "viewpoint_angular_separation_degrees"
                ),
            }

            # Carry over the reference arrow block when present, updating its
            # image_path to be project-root-relative.
            if "reference_object_arrow" in vg:
                arrow = dict(vg["reference_object_arrow"])
                if "image_path" in arrow:
                    # Arrow image lives in the same scene outputs directory.
                    arrow["image_path"] = _project_rel(
                        outputs_dir, scene_id, arrow["image_path"]
                    )
                group_record["reference_object_arrow"] = arrow

            groups.append(group_record)

            # --- Example records ---
            for vp in viewpoints:
                vi: int = vp["viewpoint_index"]
                example_id = f"{scene_id}__{pair_id}__view_{vi}"

                # metadata.json stores image paths relative to the scene dir,
                # e.g. "images/objA_10_objB_11_view_0.png".
                # Convert to a project-root-relative path.
                image_path = _project_rel(outputs_dir, scene_id, vp["image_path"])

                examples.append(
                    {
                        "example_id": example_id,
                        "group_id": group_id,
                        "scene_id": scene_id,
                        "pair_id": pair_id,
                        "viewpoint_index": vi,
                        "image_path": image_path,
                        "spatial_relations": vp["spatial_relations"],
                        "angular_sep_from_view0_deg": vp.get(
                            "angular_sep_from_view0_deg"
                        ),
                        "yaw_to_arrow": vp.get("yaw_to_arrow"),
                    }
                )

    # Write the three index files.
    _write_jsonl(index_dir / "scenes.jsonl", scenes)
    _write_jsonl(index_dir / "groups.jsonl", groups)
    _write_jsonl(index_dir / "examples.jsonl", examples)

    print(
        f"\nDataset index written to {index_dir}/\n"
        f"  {len(scenes):>6,} scenes\n"
        f"  {len(groups):>6,} groups   (object pairs)\n"
        f"  {len(examples):>6,} examples (viewpoints)\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_rel(outputs_dir: Path, scene_id: str, scene_rel_path: str) -> str:
    """Convert a scene-relative path to a project-root-relative forward-slash path.

    Example:
        outputs_dir = Path("outputs")
        scene_id    = "scene0700_00"
        scene_rel   = "images/objA_10_objB_11_view_0.png"
        → "outputs/scene0700_00/images/objA_10_objB_11_view_0.png"
    """
    return (outputs_dir / scene_id / scene_rel_path).as_posix()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write *records* to a newline-delimited JSON file."""
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(records):,} records -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a consolidated dataset index from per-scene output directories."
        )
    )
    p.add_argument(
        "--outputs_dir",
        default="outputs",
        help="Root directory containing per-scene output folders. Default: outputs/",
    )
    p.add_argument(
        "--index_dir",
        default="dataset",
        help=(
            "Directory where the consolidated index will be written. "
            "Default: dataset/"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    outputs_dir = Path(args.outputs_dir)
    index_dir = Path(args.index_dir)

    if not outputs_dir.exists():
        print(f"Error: outputs_dir not found: {outputs_dir}")
        raise SystemExit(1)

    build_index(outputs_dir, index_dir)


if __name__ == "__main__":
    main()
