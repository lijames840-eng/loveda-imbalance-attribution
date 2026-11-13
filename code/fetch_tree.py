"""Fetch the official LoveDA tree file-by-file with connection reuse (curl --parallel).

Why not one big zip: the mirror serves a freshly requested 4 GB zip at ~50 KB/s,
while many small cached PNGs over reused connections reach ~0.9 MB/s.

Usage:
    python code/fetch_tree.py --root data/raw/LoveDA --subsets Train/Urban Train/Rural
"""
from paths import DATA_RAW, TREE_CFG  # noqa: E402
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_loveda import ENDPOINT, REPO, api_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_RAW))
    ap.add_argument("--subsets", nargs="+", default=["Train/Urban", "Train/Rural"])
    ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--cfg", default=str(TREE_CFG))
    args = ap.parse_args()

    todo = []
    for sub in args.subsets:
        for kind in ("images_png", "masks_png"):
            for path, size in api_list(f"{sub}/{kind}"):
                dst = os.path.join(args.root, path.replace("/", os.sep))
                if os.path.exists(dst) and os.path.getsize(dst) == size:
                    continue
                todo.append((path, dst))
    print(f"{len(todo)} files pending", flush=True)

    os.makedirs(os.path.dirname(args.cfg), exist_ok=True)
    with open(args.cfg, "w") as f:
        for path, dst in todo:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f.write(f'url = "{ENDPOINT}/datasets/{REPO}/resolve/main/{path}"\n')
            f.write("location\nfail\nsilent\nshow-error\n")
            # curl config treats backslash as an escape character -> always use forward slashes
            f.write(f'output = "{dst.replace(os.sep, "/")}"\n')

    cmd = ["curl", "--ssl-no-revoke", "--parallel", "--parallel-max", str(args.parallel),
           "--retry", "3", "--retry-delay", "2",
           "--max-time", "180", "--speed-time", "30", "--speed-limit", "2048",
           "-K", args.cfg]
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    print("curl rc =", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
