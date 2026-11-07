"""Resumable ranged download of the Zenodo images archive."""
import hashlib
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW
PARTS = RAW / "images_parts"
URL = "https://zenodo.org/records/5166987/files/images.zip?download=1"
SIZE = 12389751014
MD5 = "510367ec6dbac77d40f77c58673a1026"
N = 8
STEP = (SIZE + N - 1) // N
lock = threading.Lock()


def fetch_part(index):
    lo, hi = index * STEP, min(SIZE, (index + 1) * STEP) - 1
    target = PARTS / f"part{index}"
    for attempt in range(20):
        current = target.stat().st_size if target.exists() else 0
        if current == hi - lo + 1:
            return
        if current > hi - lo + 1:
            raise RuntimeError(f"part {index} is oversized")
        start = lo + current
        try:
            with requests.get(URL, headers={"Range": f"bytes={start}-{hi}", "User-Agent": "WikiChurches-research/1.0"}, stream=True, timeout=120) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise RuntimeError(f"Unexpected range response: {response.status_code} {response.headers.get('Content-Range')}")
                with target.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        output.write(chunk)
        except (requests.RequestException, OSError) as exc:
            with lock:
                print(f"part {index} retry {attempt + 1}: {exc}", flush=True)
            time.sleep(min(60, 2 ** min(attempt, 5)))
    current = target.stat().st_size if target.exists() else 0
    if current != hi - lo + 1:
        raise RuntimeError(f"part {index} incomplete: {current}/{hi - lo + 1}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    first = PARTS / "part0"
    old = RAW / "images.zip"
    if old.exists() and not first.exists() and old.stat().st_size < SIZE:
        old.rename(first)
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(fetch_part, i) for i in range(N)]
        while not all(f.done() for f in futures):
            done = sum((PARTS / f"part{i}").stat().st_size for i in range(N) if (PARTS / f"part{i}").exists())
            print(f"downloaded {done}/{SIZE} bytes ({done/SIZE:.1%})", flush=True)
            time.sleep(20)
        for future in futures:
            future.result()
    archive = RAW / "images.zip"
    digest = hashlib.md5()
    with archive.open("wb") as output:
        for i in range(N):
            with (PARTS / f"part{i}").open("rb") as source:
                while chunk := source.read(16 * 1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
    actual = digest.hexdigest()
    print(f"images.zip md5={actual} expected={MD5}", flush=True)
    if actual != MD5:
        raise RuntimeError("MD5 mismatch; retained all parts for diagnosis")
    for i in range(N):
        (PARTS / f"part{i}").unlink()
    PARTS.rmdir()


if __name__ == "__main__":
    main()
