"""Download LoveDA (official split) from the HuggingFace mirror.

Layout on the mirror (repo: ahsennazir/loveDA):
    Train/Urban/{images_png,masks_png}, Train/Rural/..., Val/Urban/..., Val/Rural/...

Usage:
    python code/download_loveda.py --root data/raw/LoveDA --workers 12
"""
from paths import DATA_RAW  # noqa: E402
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "ahsennazir/loveDA"
ENDPOINT = "https://hf-mirror.com"
SUBSETS = [
    "Train/Urban", "Train/Rural", "Val/Urban", "Val/Rural",
]
KINDS = ["images_png", "masks_png"]
UA = {"User-Agent": "Mozilla/5.0 (loveDA-course-project)"}


def _open(url, timeout=60, retries=6):
    """Open a URL with exponential backoff (mirror returns 429 under load)."""
    delay = 3.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 502, 500) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 1.8, 40)
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 1.8, 40)
                continue
            raise


def api_list(path):
    """List all files under a repo path, following HF's cursor pagination."""
    files, url = [], f"{ENDPOINT}/api/datasets/{REPO}/tree/main/{urllib.parse.quote(path)}?recursive=false&limit=1000"
    while url:
        with _open(url) as r:
            items = json.load(r)
            link = r.headers.get("Link", "")
        for it in items:
            if it["type"] == "file":
                files.append((it["path"], it.get("size", 0)))
        url = None
        if 'rel="next"' in link:
            next_url = link.split("<")[1].split(">")[0]
            url = urllib.parse.urljoin(ENDPOINT, next_url)
        time.sleep(0.5)
    return files


def download(rel_path, size, root, retries=3):
    dst = os.path.join(root, rel_path.replace("/", os.sep))
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "skip", size
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{urllib.parse.quote(rel_path)}"
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            with _open(url, timeout=120, retries=4) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) == size:
                os.replace(tmp, dst)
                return "ok", size
            raise IOError(f"size mismatch {os.path.getsize(tmp)} != {size}")
        except Exception as e:
            if attempt == retries - 1:
                return f"fail:{e}", 0
            time.sleep(3 * (attempt + 1))
    return "fail", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_RAW))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    print("[1/3] listing files on mirror ...", flush=True)
    manifest = []
    for sub in SUBSETS:
        for kind in KINDS:
            path = f"{sub}/{kind}"
            files = api_list(path)
            manifest.extend(files)
            print(f"  {path:26s} {len(files):5d} files", flush=True)
    total = sum(s for _, s in manifest)
    print(f"[2/3] total {len(manifest)} files, {total / 1e9:.2f} GB", flush=True)

    meta_dir = os.path.dirname(os.path.normpath(args.root))
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "manifest_loveda.json"), "w") as f:
        json.dump([{"path": p, "size": s} for p, s in manifest], f, indent=1)

    print(f"[3/3] downloading with {args.workers} workers ...", flush=True)
    done_bytes, counts = 0, {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, p, s, args.root): p for p, s in manifest}
        for i, fut in enumerate(as_completed(futs), 1):
            status, n = fut.result()
            done_bytes += n
            key = status if status in counts else "fail"
            counts[key] += 1
            if i % 200 == 0 or i == len(futs):
                el = time.time() - t0
                rate = done_bytes / max(el, 1) / 1e6
                eta = (total - done_bytes) / max(done_bytes / max(el, 1), 1) / 60
                print(f"  {i:5d}/{len(futs)}  {done_bytes/1e9:.2f}/{total/1e9:.2f} GB  "
                      f"{rate:.2f} MB/s  eta {eta:.1f} min  {counts}", flush=True)
    print(f"finished in {(time.time()-t0)/60:.1f} min -> {counts}")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
