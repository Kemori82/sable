"""Download every model file listed in models.json into models/, checking each one's SHA-256.

Standard library only, so it runs before anything else is installed. Each file is fetched in 32 MB pieces
over 8 connections at once, so one slow connection can't hold up the install: a piece that stalls, or runs
at under a quarter of the other connections' average, is dropped and fetched again (a line that is slow
overall is left alone). Files that are already there and match are skipped, and an
interrupted download resumes from the pieces it already has. Writes models/.complete when everything matches.

Usage:  python install_models.py
"""
import hashlib
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PIECE = 32 << 20
WORKERS = 8
SLOW = 0.25  # a piece at under this fraction of the average per-connection speed (after its first 15 s) is retried
TRIES = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def url(entry: dict, manifest: dict) -> str:
    if "hf" in entry:
        hf = entry["hf"]
        return f"https://huggingface.co/{hf['repo']}/resolve/{hf['revision']}/{hf['file']}"
    return f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}/{entry['asset']}"


class Progress:
    def __init__(self, total: int):
        self.total, self.done, self.lock, self.last = total, 0, threading.Lock(), 0.0
        self.window = [(time.monotonic(), 0)]  # (time, bytes done) over the last few seconds

    def speed(self) -> float:
        """Bytes/s over the last ~5 s, all connections together."""
        (t0, d0), (t1, d1) = self.window[0], self.window[-1]
        return (d1 - d0) / (t1 - t0) if t1 - t0 > 1 else 0.0

    def add(self, n: int):
        with self.lock:
            self.done += n
            now = time.monotonic()
            if now - self.last > 0.5 or self.done >= self.total:
                self.last = now
                self.window.append((now, self.done))
                while now - self.window[0][0] > 5:
                    self.window.pop(0)
                print(f"\r  {self.done / 2**20:7.0f} / {self.total / 2**20:.0f} MB  ({self.speed() / 2**20:5.1f} MB/s)  ",
                      end="", flush=True)


def fetch_piece(src: str, part: Path, start: int, end: int, progress: Progress, whole: bool):
    """Write bytes [start, end) of src into part at the same offset, retrying slow or failed attempts."""
    for attempt in range(1, TRIES + 1):
        got = 0
        try:
            headers = {"User-Agent": "Sable-installer"}
            if not whole:
                headers["Range"] = f"bytes={start}-{end - 1}"
            with urllib.request.urlopen(urllib.request.Request(src, headers=headers), timeout=20) as r:
                if not whole and r.status != 206:
                    raise OSError(f"server sent status {r.status} instead of a range")
                t0 = time.monotonic()
                with part.open("r+b") as f:
                    f.seek(start)
                    while got < end - start:
                        chunk = r.read(min(1 << 20, end - start - got))
                        if not chunk:
                            raise OSError("connection closed early")
                        f.write(chunk)
                        got += len(chunk)
                        progress.add(len(chunk))
                        elapsed = time.monotonic() - t0
                        if elapsed > 15 and got / elapsed < SLOW * progress.speed() / WORKERS:
                            raise OSError("much slower than the other connections")
            return
        except OSError:
            progress.add(-got)
            if attempt == TRIES:
                raise
            time.sleep(attempt)


def download(entry: dict, src: str, dst: Path, progress: Progress, pool: ThreadPoolExecutor):
    size = entry["size"]
    part, done_file = dst.with_name(dst.name + ".part"), dst.with_name(dst.name + ".part.done")
    done = set(json.loads(done_file.read_text())) if part.exists() and done_file.exists() else set()
    if not part.exists() or part.stat().st_size != size:
        with part.open("wb") as f:
            f.truncate(size)
        done = set()
    starts = list(range(0, size, PIECE)) or [0]
    progress.add(sum(min(PIECE, size - s) for s in starts if s in done))
    lock = threading.Lock()

    def one(s):
        fetch_piece(src, part, s, min(s + PIECE, size), progress, whole=len(starts) == 1)
        with lock:
            done.add(s)
            done_file.write_text(json.dumps(sorted(done)))

    for fut in [pool.submit(one, s) for s in starts if s not in done]:
        fut.result()
    part.replace(dst)
    done_file.unlink(missing_ok=True)


def main() -> int:
    manifest = json.loads((ROOT / "models.json").read_text())
    files = manifest["files"]
    todo = []
    for e in files:
        dst = MODELS / e["path"]
        if not (dst.exists() and dst.stat().st_size == e["size"] and sha256(dst) == e["sha256"]):
            todo.append(e)
    have = sum(e["size"] for e in files) - sum(e["size"] for e in todo)
    print(f"Models: {len(files)} files, {len(files) - len(todo)} already there"
          + (f", downloading {sum(e['size'] for e in todo) / 2**30:.1f} GB" if todo else ""))
    progress = Progress(sum(e["size"] for e in todo))
    failed = []
    # every file's pieces share one pool of connections, so small files and big files download side by side
    with ThreadPoolExecutor(WORKERS) as pool, ThreadPoolExecutor(WORKERS) as files_pool:
        def get(e):
            dst = MODELS / e["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                download(e, url(e, manifest), dst, progress, pool)
                if sha256(dst) != e["sha256"]:
                    dst.unlink()
                    raise OSError("checksum did not match")
            except OSError as err:
                failed.append(f"{e['path']}: {err}")

        list(files_pool.map(get, todo))
    print()
    if failed:
        print("Could not download:\n  " + "\n  ".join(failed))
        print("Check the internet connection and run the installer again; it keeps what it already has.")
        return 1
    (MODELS / ".complete").write_text(manifest["tag"])
    print(f"All models downloaded and checked ({have / 2**30:.1f} GB were already there).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
