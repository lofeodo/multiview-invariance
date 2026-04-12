"""
run_chatgpt_benchmark.py

Bridge script between the generated multiview dataset and the OpenAI API.

It loads grouped viewpoint examples from dataset/, sends the group's images
plus a question to the model, and writes one JSONL record per processed group.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from chatgpt_api import ChatGPTVisionClient
from dataset import MultiviewDataset


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT = Path("results/chatgpt_results.jsonl")


def build_group_prompt(
    question: str,
    group: dict,
    examples: list[dict],
    include_arrow_view: bool,
) -> str:
    """Build a simple prompt with the minimum dataset context needed."""
    object_a = group["object_A"]
    object_b = group["object_B"]

    lines = [
        "You are given multiple images from the same 3D scene.",
        "The same two highlighted objects appear in every regular viewpoint image.",
        f"Object A: {object_a['label']} (color: {object_a['color']})",
        f"Object B: {object_b['label']} (color: {object_b['color']})",
        f"Regular viewpoint images: {len(examples)}",
        "The regular images are ordered by viewpoint_index starting at 0.",
    ]

    if include_arrow_view and "reference_object_arrow" in group:
        arrow = group["reference_object_arrow"]
        lines.append(
            "An additional final image may be included from the reference arrow's viewpoint."
        )
        lines.append(f"Reference arrow color: {arrow.get('color', 'unknown')}")

    lines.extend(
        [
            "",
            "Answer the user's question using the images provided.",
            "If your answer depends on image order, refer to them by viewpoint_index.",
            "",
            f"Question: {question.strip()}",
        ]
    )
    return "\n".join(lines)


def iter_filtered_groups(
    ds: MultiviewDataset,
    *,
    scene_id: str | None,
    group_id: str | None,
    max_groups: int | None,
) -> Iterable[tuple[dict, list[dict]]]:
    """Yield groups after applying optional scene/group filters."""
    count = 0
    for group, examples in ds.iter_groups():
        if scene_id is not None and group["scene_id"] != scene_id:
            continue
        if group_id is not None and group["group_id"] != group_id:
            continue

        yield group, sorted(examples, key=lambda ex: ex["viewpoint_index"])
        count += 1
        if max_groups is not None and count >= max_groups:
            return


def load_question(args: argparse.Namespace) -> str:
    """Load the user's benchmark question from CLI text or a file."""
    if args.question is not None:
        return args.question
    if args.question_file is not None:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    raise ValueError("Either --question or --question-file is required.")


def load_processed_group_ids(path: Path) -> set[str]:
    """Read an existing JSONL results file so reruns can skip completed groups."""
    if not path.exists():
        return set()

    processed: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            group_id = record.get("group_id")
            if isinstance(group_id, str):
                processed.add(group_id)
    return processed


def collect_image_paths(
    group: dict,
    examples: list[dict],
    include_arrow_view: bool,
) -> list[str]:
    """Collect all image paths that should be sent for one group."""
    image_paths = [ex["image_path"] for ex in examples]
    if include_arrow_view:
        arrow = group.get("reference_object_arrow", {})
        arrow_path = arrow.get("image_path")
        if arrow_path:
            image_paths.append(arrow_path)
    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ChatGPT/OpenAI over grouped viewpoint images from the dataset."
    )
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument(
        "--question",
        help="Question to ask about each group of images.",
    )
    question_group.add_argument(
        "--question-file",
        help="Path to a UTF-8 text file containing the question/prompt.",
    )
    parser.add_argument(
        "--index-dir",
        default="dataset",
        help="Dataset index directory produced by build_dataset_index.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSONL file where model responses will be written. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional path to a text file containing higher-priority instructions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--detail",
        choices=["auto", "low", "high"],
        default="auto",
        help="Image detail level sent to the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional response token cap.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. Otherwise uses OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Only process groups from one scene, e.g. scene0000_00.",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Only process one group, e.g. scene0000_00__3_7.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional cap on how many groups to process.",
    )
    parser.add_argument(
        "--include-arrow-view",
        action="store_true",
        help="Append the arrow-view image when it exists in the dataset group.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending and resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = load_question(args)
    instructions = None
    if args.system_prompt_file is not None:
        instructions = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    ds = MultiviewDataset(args.index_dir)
    client = ChatGPTVisionClient(model=args.model, api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_group_ids = set()
    write_mode = "w" if args.overwrite else "a"
    if not args.overwrite:
        processed_group_ids = load_processed_group_ids(output_path)

    with output_path.open(write_mode, encoding="utf-8") as fh:
        for group, examples in iter_filtered_groups(
            ds,
            scene_id=args.scene_id,
            group_id=args.group_id,
            max_groups=args.max_groups,
        ):
            if group["group_id"] in processed_group_ids:
                print(f"Skipping already processed group: {group['group_id']}")
                continue

            image_paths = collect_image_paths(
                group,
                examples,
                include_arrow_view=args.include_arrow_view,
            )
            prompt = build_group_prompt(
                question=question,
                group=group,
                examples=examples,
                include_arrow_view=args.include_arrow_view,
            )

            print(f"Querying {group['group_id']} with {len(image_paths)} image(s)...")
            response = client.prompt_with_images(
                prompt=prompt,
                image_sources=image_paths,
                instructions=instructions,
                detail=args.detail,
                max_output_tokens=args.max_output_tokens,
            )

            record = {
                "group_id": group["group_id"],
                "scene_id": group["scene_id"],
                "pair_id": group["pair_id"],
                "object_A": group["object_A"],
                "object_B": group["object_B"],
                "question": question,
                "system_prompt_file": args.system_prompt_file,
                "system_prompt": instructions,
                "prompt": prompt,
                "image_paths": image_paths,
                "n_regular_viewpoints": len(examples),
                "model": response.model,
                "response_id": response.response_id,
                "usage": response.usage,
                "response_text": response.text,
                "ground_truth": [
                    {
                        "viewpoint_index": ex["viewpoint_index"],
                        "spatial_relations": ex["spatial_relations"],
                        "yaw_to_arrow": ex.get("yaw_to_arrow"),
                    }
                    for ex in examples
                ],
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
