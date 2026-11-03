"""Write models.json: every file in models/ with its size, SHA-256 and where install_models.py downloads it from.

The base Qwen model comes from Hugging Face at a pinned revision; everything else (adapters, DeBERTa,
calibrations) from this repo's GitHub release, one asset per file named by its path with "/" -> "--".
After running this, upload the files listed under "release" to the release tag (see README, Publishing).

Usage:  python tools/make_manifest.py OWNER/REPO [--tag models-v1]
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HF = {"base-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--tag", default="models-v1")
    args = p.parse_args()
    files = []
    for f in sorted(MODELS.rglob("*")):
        if not f.is_file() or f.suffix == ".bak" or f.name.startswith("."):
            continue
        rel = f.relative_to(MODELS).as_posix()
        top, _, name = rel.partition("/")
        entry = {"path": rel, "size": f.stat().st_size, "sha256": sha256(f)}
        if top in HF:
            entry["hf"] = {"repo": HF[top][0], "revision": HF[top][1], "file": name}
        else:
            entry["asset"] = rel.replace("/", "--")
        files.append(entry)
        print(f"{rel:50s} {entry['size'] / 2**20:8.1f} MB")
    (ROOT / "models.json").write_text(json.dumps({"repo": args.repo, "tag": args.tag, "files": files}, indent=1))
    print(f"{len(files)} files, {sum(e['size'] for e in files) / 2**30:.2f} GB -> models.json")


if __name__ == "__main__":
    main()
