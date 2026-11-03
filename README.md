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
| Detector | DeBERTa + RAID + papers | |
| Humanizer | Evolved (round 4) | v2 RL |
| Writer | Round 3 | SFT only |

The detector shows orange from 35% AI-like (or above 90% of human writing) and red from 70%, and highlights the
parts that read most like AI. A humanizer or writer loads the first time it is used (7-25 s). **Detect** on a
humanized or written result opens the Detector tab with that text and scores it.

The humanizer checks its own work: it writes 8 rewrites of each passage and keeps the one the detector finds
least AI-like (Settings → Check with the detector: 8 tries, 4 tries or Off). The writer takes an optional
**What it should say** box with the points to make; it follows it best together with an opening sentence.

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
