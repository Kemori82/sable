# Writing Studio

The detector, humanizer and writer in one Windows app.

## Install

1. Download this repo (**Code → Download ZIP**, then unzip it) or `git clone` it.
2. Double-click **Writing Studio.bat**.

The first run sets everything up by itself: Python 3.12, PyTorch and the other packages (in `.venv` inside the
folder; nothing is installed system-wide), and the models (about 5 GB), each checked against `models.json`.
It also adds a Desktop shortcut. If the download is interrupted, run it again and it continues where it stopped.
After that, **Writing Studio.bat** or the shortcut just starts the app.

Needs Windows 10/11 and about 10 GB of disk space. An NVIDIA GPU with 8 GB is recommended; without one the app
runs on the CPU, where detection is fine but writing and humanizing are slow.

## Use

| Tab | Default | Other choice (Settings) |
|---|---|---|
| Detector | DeBERTa + RAID + papers | 80/20 with Binoculars |
| Humanizer | Evolved (round 4) | v2 RL |
| Writer | Round 3 | SFT only |

Both detector versions use the same bands: orange from 35% AI-like (or above 90% of human writing), red from 70%.
The detector loads at start; a humanizer or writer loads the first time it is used (7-25 s). **Detect** on a
humanized or written result opens the Detector tab with that text and scores it. Everything runs locally;
nothing you type leaves the computer.

## What is in models/

- `base-1.5b` Qwen2.5-1.5B-Instruct. Every writer and humanizer is this base with its chain of LoRA adapters
  merged in at load time (`adapters/`, 71 MB each), which gives bit-for-bit the trained models. It is also
  Binoculars' performer.
- `base-0.5b` Qwen2.5-0.5B-Instruct, Binoculars' observer (80/20 version only).
- `deberta` the DeBERTa + RAID + papers detector; matrix weights stored in bfloat16 (they run in bfloat16 on
  the GPU, so scores are identical), embeddings in float32.
- `calibration_*.json` the two detector versions.

The base models download from Hugging Face; the rest from this repo's `models-v1` release.

Code: `app.py` (window), `server.py` (API), `detector.py`, `generator.py`, `page.html`.
Setup: `Writing Studio.bat` → `install.ps1` → `install_models.py`.

## Publishing new models

Put the new files in `models/`, run `python tools/make_manifest.py Kemori82/writing-studio --tag models-v2`,
upload every file with an `asset` entry in `models.json` to a release with that tag (asset name = path with
`/` replaced by `--`), and commit `models.json`.
