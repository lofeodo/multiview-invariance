import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--scenes", type=int, nargs="+", metavar="ID", help="Scene IDs to download (e.g. --scenes 0 1 2)")
group.add_argument("--upto", type=int, metavar="N", help="Download first N scenes (e.g. --upto 10)")
args = parser.parse_args()

ids = args.scenes if args.scenes is not None else range(args.upto)
scenes = [f"scene{str(i).zfill(4)}_00" for i in ids]

file_suffixes = [
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json",
    ".txt"
]

patterns = [
    f"{scene}/{scene}{suffix}"
    for scene in scenes
    for suffix in file_suffixes
]

print(f"Downloading {len(scenes)} scene(s): {', '.join(scenes)}")

snapshot_download(
    repo_id="zahidpichen/scannet-dataset",
    repo_type="dataset",
    local_dir="./scannet_data",
    allow_patterns=patterns
)
