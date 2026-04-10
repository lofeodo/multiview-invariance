"""
dataset.py

In-memory loader for the consolidated multiview-invariance dataset index.

The index is produced by build_dataset_index.py and lives in a directory
(default: dataset/) that contains three JSONL files:

    scenes.jsonl    — one record per scene
    groups.jsonl    — one record per object pair within a scene (a "group")
    examples.jsonl  — one record per rendered viewpoint image (an "example")

Terminology
-----------
example  : one rendered viewpoint image for a specific object pair.
group    : all viewpoints belonging to the same object pair in the same scene.
           Viewpoints within a group must always stay together (they are the
           "unit of comparison" for the benchmark).
scene    : one ScanNet scene, which may contain several groups.

Quick start
-----------
    from dataset import MultiviewDataset

    ds = MultiviewDataset("dataset")
    print(ds)
    # MultiviewDataset(700 scenes, 2100 groups, 4200 examples)

    # Iterate examples in group order (all viewpoints of each group are adjacent)
    for example in ds.iter_examples():
        print(example["example_id"], example["image_path"])

    # Iterate groups explicitly
    for group, examples in ds.iter_groups():
        print(group["group_id"], "→", len(examples), "viewpoints")

    # All viewpoints for one pair
    examples = ds.get_group_examples("scene0700_00__10_11")

    # Shuffle groups (viewpoints within each group stay together)
    ds.shuffle_groups(seed=42)

    # Train / test split: split unit is the *scene*, so no scene leaks
    train_ds, test_ds = ds.split_by_scene(train_frac=0.8, seed=42)
    print(len(train_ds.scenes), "train scenes /", len(test_ds.scenes), "test scenes")
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultiviewDataset:
    """In-memory dataset loaded from a consolidated index directory.

    Parameters
    ----------
    index_dir : str or Path
        Directory produced by ``build_dataset_index.py``.  Must contain
        ``scenes.jsonl``, ``groups.jsonl``, and ``examples.jsonl``.
    """

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)

        # Load all three tables from disk.
        self.scenes: list[dict] = _load_jsonl(self.index_dir / "scenes.jsonl")
        self.groups: list[dict] = _load_jsonl(self.index_dir / "groups.jsonl")
        self.examples: list[dict] = _load_jsonl(self.index_dir / "examples.jsonl")

        # Fast lookup: group_id → ordered list of examples for that group.
        self._group_examples: dict[str, list[dict]] = {}
        for ex in self.examples:
            self._group_examples.setdefault(ex["group_id"], []).append(ex)

        # Fast lookup: scene_id → list of groups belonging to that scene.
        self._scene_groups: dict[str, list[dict]] = {}
        for g in self.groups:
            self._scene_groups.setdefault(g["scene_id"], []).append(g)

    # ------------------------------------------------------------------
    # Counts and display
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of examples (viewpoints) in this dataset."""
        return len(self.examples)

    def __repr__(self) -> str:
        return (
            f"MultiviewDataset("
            f"{len(self.scenes)} scenes, "
            f"{len(self.groups)} groups, "
            f"{len(self.examples)} examples)"
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_examples(self) -> Iterator[dict]:
        """Yield each example (one viewpoint image) in the current order.

        The ordering respects ``shuffle_groups``: examples for the same group
        are always adjacent, but the group order may have been shuffled.
        """
        yield from self.examples

    def iter_groups(self) -> Iterator[tuple[dict, list[dict]]]:
        """Yield ``(group, examples)`` pairs in the current group order.

        All viewpoints for the same object pair are yielded together, so the
        benchmark can always treat each group atomically.
        """
        for group in self.groups:
            yield group, self._group_examples.get(group["group_id"], [])

    def get_group_examples(self, group_id: str) -> list[dict]:
        """Return all examples (viewpoints) belonging to *group_id*.

        Returns an empty list if the group ID is not found.
        """
        return self._group_examples.get(group_id, [])

    # ------------------------------------------------------------------
    # Shuffling
    # ------------------------------------------------------------------

    def shuffle_groups(self, seed: int | None = None) -> "MultiviewDataset":
        """Shuffle the order of groups (and their viewpoints) in place.

        Viewpoints *within* each group are never reordered relative to each
        other.  Only the order in which groups appear changes.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.  Pass ``None`` for a random shuffle.

        Returns
        -------
        self
            Allows chaining: ``ds.shuffle_groups(42).iter_examples()``.
        """
        rng = random.Random(seed)
        rng.shuffle(self.groups)
        # Rebuild the flat example list to match the new group order.
        self.examples = [
            ex
            for g in self.groups
            for ex in self._group_examples.get(g["group_id"], [])
        ]
        return self

    # ------------------------------------------------------------------
    # Train / test splitting
    # ------------------------------------------------------------------

    def split_by_scene(
        self,
        train_frac: float = 0.8,
        seed: int = 42,
    ) -> tuple["MultiviewDataset", "MultiviewDataset"]:
        """Split the dataset into train and test subsets at the *scene* level.

        The split unit is the scene: every group (object pair) and every
        example (viewpoint) from a given scene ends up entirely in either the
        train set or the test set, never split across both.  This prevents any
        form of leakage between splits.

        The behaviour is analogous to ``sklearn.model_selection.GroupShuffleSplit``
        where the group key is ``scene_id``.

        Parameters
        ----------
        train_frac : float
            Fraction of scenes to place in the training set.
            Must satisfy ``0 < train_frac < 1``.
        seed : int
            Random seed used to shuffle scene IDs before splitting.

        Returns
        -------
        (train_dataset, test_dataset) : tuple[MultiviewDataset, MultiviewDataset]
            Two non-overlapping ``MultiviewDataset`` instances sharing no
            scenes, groups, or examples.

        Example
        -------
            train_ds, test_ds = ds.split_by_scene(train_frac=0.8, seed=42)
            # All examples in train_ds come from different scenes than test_ds.
        """
        if not (0 < train_frac < 1):
            raise ValueError("train_frac must be strictly between 0 and 1")

        rng = random.Random(seed)
        scene_ids = [s["scene_id"] for s in self.scenes]
        rng.shuffle(scene_ids)

        n_train = max(1, int(len(scene_ids) * train_frac))
        train_scenes = set(scene_ids[:n_train])
        test_scenes = set(scene_ids[n_train:])

        return (
            self._subset_by_scenes(train_scenes),
            self._subset_by_scenes(test_scenes),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subset_by_scenes(self, scene_ids: set[str]) -> "MultiviewDataset":
        """Return a new MultiviewDataset containing only *scene_ids*.

        Uses ``object.__new__`` to bypass ``__init__`` (which would try to
        read files from disk) and directly assigns the filtered data.
        """
        sub: MultiviewDataset = object.__new__(MultiviewDataset)
        sub.index_dir = self.index_dir
        sub.scenes = [s for s in self.scenes if s["scene_id"] in scene_ids]
        sub.groups = [g for g in self.groups if g["scene_id"] in scene_ids]
        sub.examples = [e for e in self.examples if e["scene_id"] in scene_ids]
        sub._group_examples = {
            gid: exs
            for gid, exs in self._group_examples.items()
            if exs and exs[0]["scene_id"] in scene_ids
        }
        sub._scene_groups = {
            sid: gs
            for sid, gs in self._scene_groups.items()
            if sid in scene_ids
        }
        return sub


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a newline-delimited JSON file into a list of dicts.

    Raises ``FileNotFoundError`` with a clear message if the file is absent so
    users know to run ``build_dataset_index.py`` first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset index file not found: {path}\n"
            "Run 'python build_dataset_index.py' to generate the index."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
