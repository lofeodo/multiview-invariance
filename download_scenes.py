import argparse
from huggingface_hub import snapshot_download, list_repo_files

REPO_ID = "zahidpichen/scannet-dataset"

parser = argparse.ArgumentParser()
scenes_group = parser.add_mutually_exclusive_group(required=False)
scenes_group.add_argument("--scenes", type=int, nargs="+", metavar="ID", help="Scene IDs to download (e.g. --scenes 0 1 2)")
scenes_group.add_argument("--upto", type=int, metavar="N", help="Download first N scenes, i.e. scenes 0 through N-1 (e.g. --upto 10)")
scenes_group.add_argument("--from", type=int, dest="from_", metavar="N", help="Download scene N and all subsequent scenes (e.g. --from 210)")
args = parser.parse_args()

if args.scenes is None and args.upto is None and args.from_ is None:
    print("No scenes specified — downloading all scenes.")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir="./scannet_data",
    )
else:
    if args.scenes is not None:
        scenes = [f"scene{str(i).zfill(4)}_00" for i in args.scenes]
    elif args.upto is not None:
        scenes = [f"scene{str(i).zfill(4)}_00" for i in range(args.upto)]
    else:
        # --from N: query the repo for which scenes actually exist at or after N,
        # so we only build patterns for real files rather than a huge speculative range.
        print(f"Querying available scenes from {args.from_} onwards...")
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        seen = set()
        scenes = []
        for f in all_files:
            top = f.split("/")[0]
            if top.startswith("scene") and top not in seen:
                seen.add(top)
                try:
                    if int(top[5:9]) >= args.from_:
                        scenes.append(top)
                except ValueError:
                    pass
        scenes.sort()
        if not scenes:
            print(f"No scenes found with ID >= {args.from_}.")
            raise SystemExit(0)

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

    if args.from_ is not None:
        print(f"Downloading {len(scenes)} scene(s) from {args.from_} onwards: {', '.join(scenes)}")
    else:
        print(f"Downloading {len(scenes)} scene(s): {', '.join(scenes)}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir="./scannet_data",
        allow_patterns=patterns
    )
