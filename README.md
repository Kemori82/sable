# Sable

An AI-writing detector for Windows. Paste a piece of writing and Sable shows how AI-like it reads, which parts
read most like AI, and how it compares with real human writing. Everything runs on your own computer: nothing
you paste is sent anywhere.

## Install

1. Download **Sable-1.2.1.zip** from this repo's [latest release](https://github.com/Kemori82/sable/releases/latest)
   and unzip it anywhere (for example into `Documents\Sable`).
2. Double-click **Sable.exe**.

The first start downloads the detector models (about 0.8 GB, once), checking each file against its SHA-256, with
progress in the window, and adds a Desktop shortcut. If the download is interrupted, start Sable again and it
carries on where it stopped. After that, Sable opens in under a second.

Needs Windows 10 or 11 and about 1.5 GB of disk space. Nothing else has to be installed: the files in the zip are
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

**Detailed results** (Settings → Detector) adds, next to the result, every passage the detector read with its own
AI-likeness: each sentence shaded by how AI-like it reads on its own, the words the detector reacted to most
underlined, and how much each sentence moves its passage's score. Point at a sentence for its numbers.

## Two detectors

Settings → Detector chooses between them; switching takes a few seconds.

- **Sonata** (the default, new in 1.1.0): trained on about twice as much text, including AI writing that was
  reworded to avoid detection, and text from more recent models. It catches far more reworded AI text.
- **Suite Lite**: the detector of Sable 1.0. It flags slightly fewer human essays, but misses most reworded AI text.

## How a score is made

1. **Cleaning.** The text is cleaned exactly as the training data was: Unicode normalized, curly quotes and dashes
   made plain, invisible characters removed, look-alike letters mapped to Latin, URLs, citations with a year and a
   trailing reference list removed, and spacing collapsed.
2. **Passages.** The text is split at sentence boundaries into passages of about 60 words (at most 120).
3. **Scoring.** Each passage is scored by a fine-tuned DeBERTa-v3 model. The document's score is the word-weighted
   average of its passages' scores, so it is always possible to see which parts pushed it up.
4. **Calibration.** That average is standardized and turned into a percentage (Platt scaling), and placed among
   the scores of human writing of the same kind. Passages are highlighted when they score higher than 98% of human
   passages.
5. **Out-of-scope check.** The text's position in the model's internal representation is compared with 20,000
   training documents. If it is further from its 10 nearest neighbours than 99% of validation documents are, Sable
   says the score is less reliable.

## How Sonata was trained

**Model.** [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base), fine-tuned for one epoch
(learning rate 2e-5, passages of up to 256 tokens) to score single passages. Every passage takes its document's
label. Training was balanced per kind of writing and label, so no one source dominates.

**Training data** (about 500,000 documents, split into passages):

| Source | What it adds |
|---|---|
| DACTYL (Thorat & Caines, 2025) | Student essays (including English-learner essays) and research abstracts, human and AI, from GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.2 and 3.3, and Mistral |
| [RAID](https://github.com/liamdugan/raid) (Dugan et al., 2024) | Books, news, poetry, recipes, Reddit, reviews, Wikipedia and abstracts, from GPT-2 to GPT-4, Llama, Mistral, MPT and Cohere |
| [DetectRL](https://github.com/NLP2CT/DetectRL) (Wu et al., 2024) | Abstracts, stories, news and reviews from ChatGPT, Claude, PaLM and Llama 2, with paraphrase, back-translation and typo attacks |
| [FAIDSet](https://huggingface.co/datasets/ngocminhta/FAIDSet) (English part) | Theses and abstracts, human and AI, including GPT-4o, Gemini 2 and DeepSeek V3/R1 |
| [MDTA](https://huggingface.co/datasets/nsp909/MDTA) / HC3 | Question answers by people and by four open models, with paraphrased versions |
| [Beemo](https://huggingface.co/datasets/toloka/beemo) | AI answers, and the same answers edited by experts and by other models |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Human sections of academic papers |
| Made for Sable | Essays, abstracts and paper sections written by Qwen 2.5 and 3, Granite 3.3, OLMo 2 and Claude on the topics of human texts in the training data, some reworded again by another model; and AI text rewritten to fool earlier versions of this detector |

Some of these sets are for research use only; Sable is a research project.

**Keeping the tests honest.** Documents are grouped before splitting: an AI text and the human text it was asked
to imitate, essays on the same topic, and near-duplicates always land in the same split. A new text sharing a
sentence with a text on the other side of the split is dropped. Some sets are never trained on at all: texts from
DeepSeek-V3, the TOEFL essays of [Liang et al. (2023)](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias), part
of the PERSUADE corpus of student essays, and texts from 2025-2026 models and commercial rewording tools
([WasItAIGenerated benchmark](https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09)).

**Calibration.** Fitted on the validation set (and half of the PERSUADE essays), so it describes the essays and
abstracts Sable is for. The flag level is set so that about 1% of human validation texts are flagged.

## How well Sonata works

On held-out texts it never trained on, at that flag level:

| Human writing wrongly flagged | |
|---|---|
| Student essays | 0.1% |
| PERSUADE student essays | 1.1% |
| Human research abstracts | 0.7% |
| Human academic paper sections | 0.2% |
| Non-native writers (TOEFL essays) | 12% |

| AI writing caught | |
|---|---|
| AI student essays | 91% |
| AI research abstracts | 95% |
| A model it never saw (DeepSeek-V3) | 99.6% |
| AI academic paper sections | 92% |
| DetectRL (older models, with attacks) | 96% |
| AI text run through commercial rewording tools | 64% |
| 2025-2026 models (GPT-5, Claude 5, Gemini 3 and others) | 36% |

## Limits

- **The newest models.** Text straight from 2025-2026 models is caught only about a third of the time, and much less
  after a rewording tool. A green score does not mean a text is human.
- **Non-native writers.** About 1 in 8 TOEFL essays by non-native writers is flagged. Treat a high score on such
  writing with extra care.
- **A score is evidence, not proof.** No detector is certain. Use the score and the highlighted parts as a reason to
  look closer, never as the only reason to accuse someone.
- **English only**, and texts under 50 words aren't scored; under 150 words the score is noisier.
