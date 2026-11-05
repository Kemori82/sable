# Sable

An AI-writing detector, humanizer and writer in one Windows app. Everything runs on your own computer; nothing
you type is sent anywhere.

## Install

1. Download this repo (**Code → Download ZIP**, then unzip it) or `git clone` it.
2. Double-click **Sable.bat**.

The first run sets everything up by itself: Python 3.12, PyTorch and the other packages (in `.venv` inside the
folder; nothing is installed system-wide), and the models (about 4 GB), each checked against `models.json`.
It also adds a Desktop shortcut. If the download is interrupted, run it again and it continues where it stopped.
After that, **Sable.bat** or the shortcut just starts the app.

Needs Windows 10/11 and about 9 GB of disk space. An NVIDIA GPU with 8 GB is recommended; without one the app
runs on the CPU, where detection is fine but writing and humanizing are slow.

## Use

| Tab | Default | Other choice (Settings) |
|---|---|---|
| Detector | DeBERTa + RAID + papers | Off (frees its graphics memory) |
| Humanizer | Writer voice | Evolved (round 4), v2 RL, Chain |
| Writer | Round 3 | SFT only |

The detector shows orange from 35% AI-like (or above 90% of human writing) and red from 70%, and highlights the
parts that read most like AI. A humanizer or writer loads the first time it is used (3-7 s). **Detect** on a
humanized or written result opens the Detector tab with that text and scores it.

The default humanizer, **Writer voice**, was trained from the SFT writer on its own passages, so it rewrites in that
writer's casual voice: on 24 held-out texts it left none red (Evolved: 21%), with a median of 4.5% AI-like, but
rephrases more loosely (meaning kept 0.74 vs 0.87). Evolved is still there for when the exact meaning matters most.

The humanizer checks its own work: it writes 8 rewrites of each passage and keeps the one the detector finds
least AI-like (Settings → Check with the detector: 8 tries, 4 tries or Off). The writer takes an optional
**What it should say** box with the points to make; it follows it best together with an opening sentence.

**Chain** rewrites each passage with v2 RL, then Writer voice, then Evolved for neat wording, with the detector
check on the Evolved step. On 6 held-out texts it scored lowest of all (median 10% AI-like, Writer voice 14%), but
three rewrites drift furthest from the original, so check that the points survived. It takes 30-45 s per essay.

The detector stays loaded while Sable is open (about 1 GB of graphics memory). Settings → Detector → Off unloads it;
the Detector tab then won't score text and the humanizer makes one try per passage, until it is turned back on.

Settings also has eight themes: System, Light, Dark, Sable, Aurora, Midnight, Liquid Glass and Space. Choices are saved in `settings.json`.

## What is in models/

- `base-1.5b` Qwen2.5-1.5B-Instruct. Every writer and humanizer is this base with its chain of LoRA adapters
  merged in at load time (`adapters/`, 71 MB each), which gives bit-for-bit the trained models.
- `deberta` the DeBERTa + RAID + papers detector; matrix weights stored in bfloat16 (they run in bfloat16 on
  the GPU, so scores are identical), embeddings in float32.
- `calibration_deberta_paper.json` turns the detector's output into a percentage, with the 35% / 70% bands.

The base model downloads from Hugging Face; the rest from this repo's `models-v1` release.

Code: `app.py` (window), `server.py` (API), `detector.py`, `generator.py`, `page.html`. Icon: `icon.svg` → `app.ico`.
Setup: `Sable.bat` → `install.ps1` → `install_models.py`.

## Publishing new models

Put the new files in `models/`, run `python tools/make_manifest.py OWNER/REPO --tag models-v2`, upload every file
with an `asset` entry in `models.json` to a release with that tag (asset name = path with `/` replaced by `--`),
and commit `models.json`.

## Updates

When Sable opens it checks GitHub for a newer version and, if there is one, offers it in a banner (**Update** or
**Later**; Settings → Updates turns the check off). Updating replaces the changed code files, updates packages or
downloads new models only if those changed, and restarts the app. In a git clone it runs `git pull` instead, and only
when there are no uncommitted changes. Anything pushed to `main` reaches installed copies this way.
