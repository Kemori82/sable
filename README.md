# Sable

An AI-writing detector for Windows. Paste a piece of writing and Sable shows how AI-like it reads, which parts
read most like AI, and how it compares with real human writing. Everything runs on your own computer: nothing
you paste is sent anywhere.

## Install

1. Download **Sable-1.0.0.zip** from this repo's [latest release](https://github.com/Kemori82/sable/releases/latest)
   and unzip it anywhere (for example into `Documents\Sable`).
2. Double-click **Sable.exe**.

The first start downloads the models (about 4.4 GB, once), checking each file against its SHA-256, with progress
in the window, and adds a Desktop shortcut. If the download is interrupted, start Sable again and it carries on
where it stopped. After that, Sable opens in under a second.

Needs Windows 10 or 11 and about 5 GB of disk space. Nothing else has to be installed: the files in the zip are
everything Sable needs. A graphics card is used when there is one (any card that supports DirectX 12: NVIDIA, AMD
or Intel); without one Sable runs on the CPU, where scoring takes a little longer. Hover over the status pill at
the top right to see what it is running on.

When a new version comes out, Sable offers it in a banner when it opens (**Update** or **Later**). Updating checks
each downloaded file and restarts the app. Settings → Updates turns the check off.

## What the detector shows

- **A score from 0 to 100% AI-like.** Orange from 35% (or when the text scores higher than 90% of human writing),
  red from 70%.
- **The parts that read most like AI**, highlighted.
- **Where the text falls among human writing**: for example "higher than 99% of human writing".
- **Notes** when a score is less reliable: short texts (under 150 words), text unlike anything the detector was
  trained on, or look-alike letters (Cyrillic or Greek letters that look Latin, which are converted before scoring).

Choose the kind of writing (Essay, Abstract or Other) before detecting; each has its own reference. Texts under
50 words, and texts that aren't English, are not scored.

## How a score is made

1. **Cleaning.** The text is cleaned exactly as the training data was: Unicode normalized, curly quotes and dashes
   made plain, invisible characters removed, look-alike letters mapped to Latin, URLs, citations with a year and a
   trailing reference list removed, and spacing collapsed.
2. **Segments.** The text is split at sentence boundaries into segments of about 60 words (at most 120).
3. **Scoring.** Each segment is scored by a fine-tuned DeBERTa-v3 model. The document's score is the
   word-weighted average of its segments' scores, so it is always possible to see which parts pushed it up.
4. **Calibration.** That average is standardized and turned into a percentage (Platt scaling), and placed among
   the scores of human writing of the same kind. Segments are highlighted when they score higher than 98% of human
   segments.
5. **Out-of-scope check.** The text's position in the model's internal representation is compared with 20,000
   training documents. If it is further from its 10 nearest neighbours than 99% of validation documents are, Sable
   says the score is less reliable.

## How the detector was trained

**Model.** [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base), fine-tuned for one epoch
(learning rate 2e-5, segments of up to 256 tokens) to score single segments. Every segment takes its document's
label. Training was balanced per kind of writing and label, so no one source dominates.

**Training data** (about 150,000 documents, split into segments):

| Source | What it adds | License |
|---|---|---|
| DACTYL (Thorat & Caines, 2025) | Student essays (including English-learner essays) and research abstracts, human and AI, from 11 models: GPT-4o and GPT-4o mini, Claude 3.5 Sonnet and Haiku, Gemini 1.5 Pro and Flash, Llama 3.2 and 3.3, Mistral Large and Small | MIT |
| [RAID](https://github.com/liamdugan/raid) (Dugan et al., 2024) | Books, news, poetry, recipes, Reddit, reviews, Wikipedia and abstracts, from GPT-2 to GPT-4, Llama, Mistral, MPT and Cohere, base and chat models | MIT |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Human sections of academic papers (2015-2021) | ODC-BY |
| Claude-written paper sections | AI sections written by Claude Sonnet and Haiku from the same papers' titles and abstracts, so the topic can't give the answer away | made for Sable |
| Hard cases | AI text deliberately rewritten to avoid detection: by humanizing tools, and by a small model trained to fool an earlier version of this detector | made for Sable |

**Keeping the tests honest.** Documents are grouped before splitting: an AI text and the human text it was asked
to imitate, essays on the same topic, RAID texts from the same source, and near-duplicates (5-word MinHash,
Jaccard over 0.8) always land in the same split (train 70%, validation 15%, test 15%). Some sets are never trained
on at all: texts from DeepSeek-V3 (a model the detector never saw), the TOEFL essays of
[Liang et al. (2023)](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias), part of the PERSUADE corpus of student
essays, and held-out RAID and paper sections.

**Avoiding a shortcut.** About 5,000 AI essays imitate native high-school writers and have no human counterpart.
They are left out of training so the model can't learn "plain, simple writing means AI", which would wrongly flag
young and non-native writers.

**Calibration.** Fitted on the validation set (and half of the PERSUADE essays), leaving RAID out so it describes
the essays and abstracts Sable is for. The flag level is set so that about 1% of human validation texts are
flagged.

## How well it works

On held-out texts the detector never trained on:

| Human writing wrongly flagged | |
|---|---|
| Student essays | 0.2% |
| PERSUADE student essays | 0.5% |
| English-learner essays | 0.0% |
| Human research abstracts | 1.0% |
| Human academic paper sections | 0.0% |
| Non-native writers (TOEFL essays) | 27.8% |

| AI writing caught | |
|---|---|
| AI student essays | 92.4% |
| AI research abstracts | 97.1% |
| A model it never saw (DeepSeek-V3) | 99.6% |
| AI academic paper sections | 95-96% |
| RAID, all 11 generators (RAID's measure: at 5% false flags) | 96.6% |
| AI text run through humanizing tools | 77.3% |

## Limits

- **Non-native writers.** About 1 in 4 TOEFL essays by non-native writers is flagged. Treat a high score on such
  writing with extra care.
- **A score is evidence, not proof.** No detector is certain. Use the score and the highlighted parts as a reason to
  look closer, never as the only reason to accuse someone.
- **Newer models.** The newest models in the training data date from late 2024, plus some Claude writing from 2026.
  Text from much newer models may be caught less often.
- **English only**, and texts under 50 words aren't scored; under 150 words the score is noisier.
