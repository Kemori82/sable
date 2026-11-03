"""AI-likeness of a text, as in the review app (aidetecor/scripts/scorer.py), trimmed to what this app uses.

Two versions share one DeBERTa model and differ in calibration:
  deberta  DeBERTa + RAID + papers                       (models/calibration_deberta_paper.json)
  mix      the same + Binoculars, 80/20 with override     (models/calibration_deberta_paper_bino.json)
Binoculars (two Qwen models) is loaded only when the 80/20 version is used.
"""
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

MODELS = Path(__file__).resolve().parent / "models"
CALIBRATIONS = {"deberta": "calibration_deberta_paper.json", "mix": "calibration_deberta_paper_bino.json"}
UNSURE_PERCENTILE = 90.0  # orange from here: scores higher than 90% of human papers

# ---- text cleaning (build_dataset.normalize): the same cleaning the model was trained on ----
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
REF_HEADING = re.compile(r"\s(References|Works Cited|Reference List|Bibliography)\s")
CITATION = re.compile(r"\((?:[^()]*?\b(?:19|20)\d{2}[a-z]?\b[^()]*?)\)")
URL = re.compile(r"https?://\S+|www\.\S+")
CONFUSABLES = str.maketrans({  # Cyrillic/Greek letters that look Latin
    "а": "a", "е": "e", "і": "i", "о": "o", "с": "c", "р": "p", "у": "y", "х": "x", "ѕ": "s", "ј": "j", "һ": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "І": "I", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X", "ο": "o"})
CONFUSABLE_CHARS = re.compile("[" + "".join(chr(k) for k in CONFUSABLES) + "]")
PLACEHOLDERS = [
    (re.compile(r"\b(?:Generic_Name|STUDENT_NAME|TEACHER_NAME|PROPER_NAME|Generic_Teacher)\b"), "Alex"),
    (re.compile(r"\b(?:Generic_City|LOCATION_NAME|Generic_Location|Generic_State)\b"), "Springfield"),
    (re.compile(r"\b(?:Generic_School|SCHOOL_NAME)\b"), "Lincoln School"),
    (re.compile(r"\bGeneric_\w+\b"), "Alex"),
]
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
LEAD_HEADING = re.compile(r"^(Introduction|Abstract)\b[:.]?\s+")
ENGLISH_WORDS = set("the of and to a in is that for it as was with be by on not are this from or have an "
                    "which at but they his her their we you can has were been more one all there would".split())


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(QUOTES)
    t = ZERO_WIDTH.sub("", t).translate(CONFUSABLES)
    m = list(REF_HEADING.finditer(t))
    if m and m[-1].start() > 0.5 * len(t):  # trailing reference list
        t = t[: m[-1].start()]
    for pattern, repl in PLACEHOLDERS:
        t = pattern.sub(repl, t)
    t = CITATION.sub("", URL.sub("", t))
    t = LEAD_HEADING.sub("", re.sub(r"\s+", " ", t).strip())
    return re.sub(r"\s+([.,;:])", r"\1", t)


# ---- segments (segments.py): sentences packed to ~60 words, max 120, a short tail merged ----
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")


def segment(text: str) -> list[tuple[int, int]]:
    spans, cursor = [], 0
    for s in (p.strip() for p in SENTENCE_END.split(text)):
        if s:
            start = text.find(s, cursor)
            spans.append((start, start + len(s)))
            cursor = start + len(s)
    out, cur, words = [], None, 0
    for start, end in spans:
        w = len(text[start:end].split())
        if cur is not None and (words >= 60 or words + w > 120):
            out.append(cur)
            cur, words = None, 0
        cur = (cur[0], end) if cur else (start, end)
        words += w
    if cur:
        if out and words < 25:
            out[-1] = (out[-1][0], cur[1])
        else:
            out.append(cur)
    return out


class Binoculars:
    """Performer log-perplexity / observer-performer cross-perplexity: low = AI-like."""

    def __init__(self, device, max_tokens: int = 512):
        self.device, self.max_tokens = device, max_tokens
        self.tok = AutoTokenizer.from_pretrained(MODELS / "base-1.5b")
        self.tok.padding_side = "right"
        self.obs = AutoModelForCausalLM.from_pretrained(MODELS / "base-0.5b", dtype=torch.bfloat16).to(device).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(MODELS / "base-1.5b", dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        enc = self.tok([text], return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
        lo, lp = self.obs(**enc).logits[0, :-1], self.perf(**enc).logits[0, :-1]
        vocab = min(lo.shape[-1], lp.shape[-1])  # the two checkpoints pad the vocabulary differently
        lo, lp = lo[:, :vocab].float(), lp[:, :vocab].float()
        logp = torch.log_softmax(lp, -1)
        ppl = -logp.gather(-1, enc.input_ids[0, 1:].unsqueeze(-1)).mean()
        xppl = -(torch.softmax(lo, -1) * logp).sum(-1).mean()
        return float(ppl / xppl)


class Detector:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cals = {k: json.loads((MODELS / f).read_text()) for k, f in CALIBRATIONS.items()}
        d = MODELS / "deberta"
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float32).to(self.device).eval()
        self.ood_ref = torch.tensor(np.load(d / "ood_reference.npy").astype(np.float32), device=self.device)
        self.ood = json.loads((d / "ood.json").read_text())
        self.bino = None

    def load_binoculars(self):
        if self.bino is None:
            self.bino = Binoculars(self.device)

    def unload_binoculars(self):
        self.bino = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _deberta(self, texts: list[str]):
        enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(**enc, output_hidden_states=True)
        return out.logits[:, 0].float().cpu().numpy(), out.hidden_states[-1][:, 0].float()

    def score(self, raw_text: str, doc_type: str = "other", version: str = "deberta") -> dict:
        """Score, band (green / orange / red), highlighted segments, each model on its own, and notes."""
        cal = self.cals[version]
        text = normalize(raw_text)
        words = text.split()
        res = {"version": version, "doc_type": doc_type, "word_count": len(words), "text": text, "notes": []}
        if len(words) < cal["min_words"]:
            return {**res, "status": "too_short", "notes": [f"Fewer than {cal['min_words']} words: too short to score."]}
        lower = [w.strip(".,;:!?\"'()").lower() for w in words]
        if sum(w in ENGLISH_WORDS for w in lower) / len(words) < 0.08:
            return {**res, "status": "unsupported_language", "notes": ["Only English text can be scored."]}

        spans = segment(text)
        seg_texts = [text[s:e] for s, e in spans]
        logits, cls = [], []
        for b in range(0, len(seg_texts), 32):
            lg, c = self._deberta(seg_texts[b:b + 32])
            logits.append(lg)
            cls.append(c)
        logits, cls = np.concatenate(logits), torch.cat(cls)
        w = np.array([len(t.split()) for t in seg_texts], float)
        std = lambda v, which, name: (np.asarray(v) - cal[which][name]["mean"]) / cal[which][name]["std"]
        deb = cal["models"]["deberta"]
        z = {"deberta": float(std((logits * w).sum() / w.sum(), "doc_standardization", deb))}
        seg_z = std(logits, "segment_standardization", deb)
        if "binoculars" in cal["models"]:
            self.load_binoculars()
            p = cal["doc_standardization"]["binoculars"]  # sign -1: low Binoculars = AI-like
            z["binoculars"] = p["sign"] * (self.bino.score(text) - p["mean"]) / p["std"]

        combine = cal.get("combine", {"type": "min"})
        s = float(sum(wt * z[k] for k, wt in combine["weights"].items())) if combine["type"] == "weighted" else min(z.values())
        sig = lambda pl, v: 1 / (1 + math.exp(-(pl["a"] * v + pl["b"])))
        prob = sig(cal["platt"], s)
        domain = doc_type if doc_type in ("essay", "abstract") else None
        threshold = cal["review_thresholds"][domain or "default"]
        ref = cal["human_percentile_reference"]
        pct = float(np.interp(s, ref["scores"][domain or "essay"], ref["grid"]))
        review = s >= threshold
        t2 = cal.get("tier2")  # second check: the most AI-like quarter of segments above every human paper
        if t2 and doc_type in t2["thresholds"]:
            k = max(1, math.ceil(len(logits) * t2["top_fraction"]))
            review = review or float(np.sort(logits)[::-1][:k].mean()) >= t2["thresholds"][doc_type]
        ov, overridden = combine.get("override"), False
        if ov and ov["model"] in z:  # DeBERTa decides alone when it is sure, or when the two disagree a lot
            po = sig(ov["platt"], z[ov["model"]])
            gap = abs(po - sig(ov["other_platt"], z[ov["other"]])) if "max_gap" in ov and ov.get("other") in z else 0.0
            if po >= ov["prob"] or gap > ov.get("max_gap", 1.0):
                prob, overridden = po, True
        bands = cal.get("bands")
        if bands:
            review = prob >= bands["review_prob"]
        band = ("red" if review else "orange" if (bands and prob >= bands["unsure_prob"]) or pct >= UNSURE_PERCENTILE
                else "green")

        comps = []
        cref = cal.get("component_reference")
        if cref and len(z) > 1:
            for key, label in (("deberta", "DeBERTa"), ("binoculars", "Binoculars")):
                m = cref["models"].get(key)
                if m and key in z and "platt" in m:
                    comps.append({"key": key, "label": label, "prob": sig(m["platt"], z[key]),
                                  "human_percentile": float(np.interp(z[key], m["percentile_scores"][domain or "essay"], cref["grid"]))})
        notes = []
        v = cls.mean(dim=0)
        if float(1 - (self.ood_ref @ (v / v.norm())).topk(self.ood["k"]).values.mean()) > self.ood["cutoff"]:
            notes.append("This text is unlike the writing the detector was trained on, so the score is less reliable.")
        if len(words) < cal["short_text_words"]:
            notes.append(f"Short text (under {cal['short_text_words']} words): scores are noisier.")
        if CONFUSABLE_CHARS.search(raw_text):
            notes.append("Look-alike characters (e.g. Cyrillic letters) were converted to Latin before scoring.")
        return {**res, "status": "scored", "score": prob, "band": band, "human_percentile": pct, "overridden": overridden,
                "unsure_from": bands["unsure_prob"] if bands else None, "review_from": bands["review_prob"] if bands else None,
                "components": comps, "notes": notes,
                "segments": [{"start": int(a), "end": int(b), "highlight": bool(v >= cal["segment_highlight_cutoff"])}
                             for (a, b), v in zip(spans, seg_z)]}
