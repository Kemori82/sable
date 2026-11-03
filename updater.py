"""Updates from GitHub: compare this folder with the repo's main branch and bring over what changed.

Two kinds of install:
  zip  downloaded with Code -> Download ZIP. Compares every file with the latest ZIP of main (line endings
       ignored) and copies over the ones that differ. Files that are only here (settings.json, models/, .venv)
       are never touched.
  git  a git clone. Uses `git pull --ff-only`, and only when there are no uncommitted changes, so a folder
       someone is working in is never overwritten.

After the files, the packages are updated if requirements.txt changed, and the models if models.json changed
(install_models.py downloads only the files that are new). Then the app restarts.
"""
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: no console flashing up from the windowed app
TEXT = {".py", ".html", ".md", ".txt", ".json", ".ps1", ".bat", ".svg", ".gitignore", ".gitattributes"}


def repo() -> str:
    return json.loads((ROOT / "models.json").read_text())["repo"]


def fetch(url: str, timeout: int = 20) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Sable-updater"}), timeout=timeout) as r:
        return r.read()


def same(path: Path, data: bytes) -> bool:
    if not path.exists():
        return False
    local = path.read_bytes()
    if path.suffix in TEXT or path.name in TEXT:  # git may have checked these out with \r\n
        return local.replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n")
    return local == data


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True,
                          creationflags=NO_WINDOW).stdout.strip()


def latest_zip() -> dict[str, bytes]:
    """{path: contents} of every file on main."""
    z = zipfile.ZipFile(io.BytesIO(fetch(f"https://codeload.github.com/{repo()}/zip/refs/heads/main", 60)))
    files = {}
    for info in z.infolist():
        rel = info.filename.partition("/")[2]  # drop the "sable-main/" folder
        if rel and not info.is_dir():
            files[rel] = z.read(info)
    return files


def check() -> dict:
    """{"available": bool, "files": [...], "message": latest commit's title, "mode": "zip" | "git", "note": ...}"""
    if (ROOT / ".git").exists():
        git("fetch", "-q", "origin", "main")
        files = [f for f in git("diff", "--name-only", "HEAD", "origin/main").splitlines() if f]
        behind = int(git("rev-list", "--count", "HEAD..origin/main") or 0)
        res = {"mode": "git", "available": behind > 0, "files": files, "message": git("log", "-1", "--format=%s", "origin/main")}
        if behind and git("status", "--porcelain", "--untracked-files=no"):
            res.update(available=False, note="This folder has changes that aren't committed, so it won't update itself. "
                                             "Commit or undo them, or run git pull.")
        return res
    files = [rel for rel, data in latest_zip().items() if not same(ROOT / rel, data)]
    try:
        message = json.loads(fetch(f"https://api.github.com/repos/{repo()}/commits/main"))["commit"]["message"].split("\n")[0]
    except (OSError, ValueError, KeyError):
        message = ""
    return {"mode": "zip", "available": bool(files), "files": files, "message": message}


def apply(say) -> None:
    """Bring this folder up to date, calling say(line) with progress. Raises on failure."""
    if (ROOT / ".git").exists():
        say("Pulling from GitHub…")
        before = git("rev-parse", "HEAD")
        git("pull", "-q", "--ff-only", "origin", "main")
        changed = set(git("diff", "--name-only", before, "HEAD").splitlines())
    else:
        say("Downloading the latest version…")
        changed = set()
        for rel, data in latest_zip().items():
            dst = (ROOT / rel).resolve()
            if ROOT not in dst.parents or same(dst, data):  # never write outside the app folder
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".new")
            tmp.write_bytes(data)
            tmp.replace(dst)
            changed.add(rel)
    say(f"Updated {len(changed)} file{'s' * (len(changed) != 1)}.")
    python = Path(sys.executable).with_name("python.exe")
    if "requirements.txt" in changed:
        say("Updating packages…")
        uv = ROOT / ".tools" / "uv.exe"
        cmd = ([str(uv), "pip", "install", "--python", str(python), "-r", "requirements.txt"] if uv.exists()
               else [str(python), "-m", "pip", "install", "-r", "requirements.txt"])
        run(cmd, say)
    if "models.json" in changed:
        say("Downloading new models…")
        run([str(python), "-u", "install_models.py"], say)


def run(cmd: list[str], say) -> None:
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         creationflags=NO_WINDOW)
    for line in p.stdout:
        line = line.rstrip().split("\r")[-1].strip()
        if line:
            say(line)
    if p.wait():
        raise RuntimeError(f"{Path(cmd[0]).name} failed (exit {p.returncode})")


def restart() -> None:
    """Start the app again the way it was started, then end this process (closing its window)."""
    subprocess.Popen([sys.executable, *sys.argv], cwd=ROOT, creationflags=0x00000008)  # DETACHED_PROCESS
    os._exit(0)
