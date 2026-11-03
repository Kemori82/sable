"""Download every model file listed in models.json into models/, checking each one's SHA-256.

Standard library only, so it runs before anything else is installed. Files that are already there and match
are skipped; an interrupted download resumes where it stopped. Writes models/.complete when everything matches.

Usage:  python install_models.py
"""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


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


def download(src: str, dst: Path, size: int, label: str):
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(src, headers={"User-Agent": "WritingStudio-installer",
                                               **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        with part.open("ab" if have else "wb") as f:
            done, last = have, 0.0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 0.5 or done == size:
                    last = time.monotonic()
                    print(f"\r  {label}: {done / 2**20:7.0f} / {size / 2**20:.0f} MB", end="", flush=True)
    print()
    part.replace(dst)


def main() -> int:
    manifest = json.loads((ROOT / "models.json").read_text())
    files = manifest["files"]
    total = sum(e["size"] for e in files)
    print(f"Models: {len(files)} files, {total / 2**30:.1f} GB")
    for i, e in enumerate(files, 1):
        dst = MODELS / e["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        label = f"[{i}/{len(files)}] {e['path']}"
        if dst.exists() and dst.stat().st_size == e["size"] and sha256(dst) == e["sha256"]:
            print(f"  {label}: already there")
            continue
        for attempt in range(1, 4):
            try:
                download(url(e, manifest), dst, e["size"], label)
                if sha256(dst) == e["sha256"]:
                    break
                dst.unlink()
                print(f"  {label}: checksum did not match, downloading again")
            except OSError as err:
                print(f"\n  {label}: {err} (attempt {attempt} of 3)")
                time.sleep(3 * attempt)
        else:
            print(f"Could not download {e['path']}. Check the internet connection and run the installer again.")
            return 1
    (MODELS / ".complete").write_text(manifest["tag"])
    print("All models downloaded and checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
