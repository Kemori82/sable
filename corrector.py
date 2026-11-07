"""Corrector: takes the original text and its humanized version, puts back meaning that was lost, and fixes grammar,
without undoing the humanizing.

  0. Repeats   a phrase repeated over and over is cut to one, and repeated sentences are dropped.
  1. Meaning   an NLI model (cross-encoder/nli-deberta-v3-base, Apache-2.0) checks each paragraph both ways:
               an original sentence the humanized paragraph doesn't support is missing; a humanized sentence the
               original contradicts (a flipped claim, which similarity scores miss) or gives no support at all
               (made up) is wrong.
  2. Repair    sentence by sentence (repair): a wrong sentence is taken out, a missing point goes back in as one new
               sentence humanized on its own. Each change is kept only if it lowers the problem count and keeps the
               paragraph under the detector limit; everything else stays as written. (gentle=False: the old way,
               re-humanizing whole flagged paragraphs, which changed far more.)
  3. Grammar   a small T5 grammar model (Unbabel/gec-t5_small, Apache-2.0, 0.12 GB) fixes each sentence; changes
               that are only spacing are undone. A fix is kept only if it is small (most words unchanged) and the
               paragraph's detector score doesn't rise past the line. (CoEdIT-large, 1.57 GB and non-commercial, found
               more fixes but made no difference to meaning or detector scores in testing.) Optional: Sable's tuned
               copy (models/gec-tuned) with safe_check, which puts back changes that aren't grammar-sized.

Returns the corrected text and a word-level diff of what changed.
"""
import difflib
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

import devices

MODELS = Path(__file__).resolve().parent / "models"
# Sable's copies (bfloat16) when installed, otherwise downloaded from Hugging Face
NLI_NAME = str(MODELS / "nli") if (MODELS / "nli").exists() else "cross-encoder/nli-deberta-v3-base"
GEC_NAME = str(MODELS / "gec") if (MODELS / "gec").exists() else "Unbabel/gec-t5_small"
# Sable's tuned grammar fixer (Settings -> Correct: grammar fixer -> Tuned): the same model trained further on
# made-up errors and on realistic ones from Google's C4_200M (CC BY 4.0); used with safe_check on
GEC_TUNED = str(MODELS / "gec-tuned") if (MODELS / "gec-tuned" / "config.json").exists() else None
GEC_PREFIX = "gec: "
MISSING, CONTRA = 0.5, 0.5   # entailment below MISSING = missing point; contradiction above CONTRA = wrong
MADE_UP = 0.05               # a humanized sentence the original supports less than this is made up
LINE = 0.35                  # the detector's orange line


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(*\[])", text.strip()) if s]


def unrepeat(text: str) -> str:
    """A run of 1-8 words repeated 3+ times in a row cut to one; a sentence seen before (same words) dropped."""
    text = re.sub(r"((?:\S+[ \t]+){1,8}?)\1{2,}", r"\1", text + " ").strip()
    out, seen = [], set()
    for para in paragraphs(text):
        keep = []
        for s in sentences(para):
            key = re.sub(r"\W+", " ", s.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                keep.append(s)
        if keep:
            out.append(" ".join(keep))
    return "\n\n".join(out)


class Corrector:
    def __init__(self, device: str | None = None, nli: str = NLI_NAME, gec: str = GEC_NAME, gentle: bool = True,
                 gec_prefix: str = GEC_PREFIX, gec_tokenizer: str | None = None, safe_check: bool = False):
        self.gentle = gentle
        self.safe_check = safe_check  # put back grammar changes that aren't grammar-sized (safe_change)
        self.gec_prefix = gec_prefix
        self.grammar_fixes = []  # (before, after) of the last correct() call's kept grammar fixes  # sentence-level repair (repair); False = re-humanize flagged paragraphs (patch + best)
        self.device = device or devices.best()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.ntok = AutoTokenizer.from_pretrained(nli)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli, dtype=dtype).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.nli_model.config.id2label.items()}
        self.ENT, self.CON = labels["entailment"], labels["contradiction"]
        self.gtok = AutoTokenizer.from_pretrained(gec_tokenizer or gec)
        self.gec_model = AutoModelForSeq2SeqLM.from_pretrained(gec, dtype=dtype).to(self.device).eval()

    # ---- NLI ----
    @torch.no_grad()
    def nli(self, premises: list[str], hypotheses: list[str]) -> torch.Tensor:
        """Probabilities [n, 3] for each (premise, hypothesis) pair."""
        out = []
        for i in range(0, len(premises), 16):
            enc = self.ntok(premises[i:i + 16], hypotheses[i:i + 16], truncation=True, max_length=512, padding=True,
                            return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.nli_model(**enc).logits.float(), -1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 3)

    def check(self, orig: str, hum: str) -> dict:
        """Which original sentences hum leaves out, and which of hum's sentences the original contradicts."""
        o, h = sentences(orig), sentences(hum)
        p = self.nli([hum] * len(o) + [orig] * len(h), o + h)
        ent, con, sup = p[:len(o), self.ENT].tolist(), p[len(o):, self.CON].tolist(), p[len(o):, self.ENT].tolist()
        return {"orig": o, "hum": h, "missing": [i for i, e in enumerate(ent) if e < MISSING],
                "wrong": [j for j in range(len(h)) if con[j] > CONTRA or sup[j] < MADE_UP]}

    @staticmethod
    def problems(c: dict) -> int:
        return len(c["missing"]) + len(c["wrong"])

    # ---- repair ----
    @staticmethod
    def patch(c: dict) -> str:
        """hum's sentences without the wrong ones, each missing original sentence put back at its relative place."""
        keep = [s for j, s in enumerate(c["hum"]) if j not in c["wrong"]]
        n_o = max(1, len(c["orig"]))
        slots = [(j / max(1, len(keep)), 1, s) for j, s in enumerate(keep)]
        slots += [(i / n_o, 0, c["orig"][i]) for i in c["missing"]]
        return " ".join(s for _, _, s in sorted(slots))

    def best(self, target: str, orig_para: str, cands: list[str], probs: list[float]) -> str:
        """The candidate that contradicts nothing in the original and supports the most of target's sentences,
        preferring those under the detector line."""
        t = sentences(target)
        if not t:
            return cands[0]
        p = self.nli([c for c in cands for _ in t], t * len(cands)).view(len(cands), len(t), 3)
        cover = p[:, :, self.ENT].mean(1).tolist()
        q, made_up = [], []
        for c in cands:  # against the original: the worst contradiction, and the share of sentences it doesn't support
            cs = sentences(c) or [c]
            r = self.nli([orig_para] * len(cs), cs)
            q.append(float(r[:, self.CON].max()))
            made_up.append(float((r[:, self.ENT] < MADE_UP).float().mean()))
        key = lambda i: (q[i] < CONTRA, probs[i] < LINE, cover[i] - made_up[i] - 0.5 * max(0.0, probs[i] - LINE))
        return cands[max(range(len(cands)), key=key)]

    def repair(self, o: str, h: str, c: dict, rewrite, detect) -> tuple[str, dict]:
        """Sentence-level repair: the rest of the paragraph is left as it is.
        Each wrong sentence is taken out, and each missing point goes back in as one new sentence (the original
        sentence humanized on its own, or as written if no try keeps its meaning), at its place in the paragraph.
        A change stays only if it lowers the problem count and the paragraph stays under the detector limit."""
        s, orig, probs = list(c["hum"]), c["orig"], self.problems(c)
        limit = max(LINE, detect([h])[0] + 0.05)
        n = {"restored": 0, "removed": 0}
        for j in sorted(c["wrong"], reverse=True):
            trial = s[:j] + s[j + 1:]
            if not trial:
                continue
            p = self.problems(self.check(o, " ".join(trial)))
            if p < probs and detect([" ".join(trial)])[0] <= limit:
                s, probs = trial, p
                n["removed"] += 1
        for i in self.check(o, " ".join(s))["missing"]:
            cands = [x for x in rewrite(orig[i]) if x.strip()] + [orig[i]]
            keeps = self.nli(cands, [orig[i]] * len(cands))[:, self.ENT].tolist()
            pos = round(i / max(1, len(orig)) * len(s))
            trials = [s[:pos] + [x] + s[pos:] for x in cands]
            ai = detect([" ".join(t) for t in trials])
            for k in sorted(range(len(cands)), key=lambda k: (keeps[k] < MISSING, ai[k]))[:3]:
                p = self.problems(self.check(o, " ".join(trials[k])))
                if p < probs and ai[k] <= limit:
                    s, probs = trials[k], p
                    n["restored"] += 1
                    break
        return " ".join(s), n

    # ---- grammar ----
    @torch.no_grad()
    def fix_grammar(self, sents: list[str]) -> list[str]:
        out = []
        for i in range(0, len(sents), 16):
            enc = self.gtok([self.gec_prefix + s for s in sents[i:i + 16]], truncation=True, max_length=256, padding=True,
                            return_tensors="pt").to(self.device)
            gen = self.gec_model.generate(**enc, max_new_tokens=160, num_beams=4)
            out += [g.strip() for g in self.gtok.batch_decode(gen, skip_special_tokens=True)]
        return [respace(s, f) for s, f in zip(sents, out)]

    @staticmethod
    def small_edit(a: str, b: str) -> bool:
        wa, wb = a.split(), b.split()
        return difflib.SequenceMatcher(None, wa, wb).ratio() >= 0.8 and len(wb) <= len(wa) + 4

    @staticmethod
    def safe(a: str, b: str) -> str:
        """b with each change that isn't a grammar-sized fix put back to a (see safe_change)."""
        wa, wb, out = a.split(), b.split(), []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
            keep = tag == "equal" or safe_change(wa[i1:i2], wb[j1:j2], wa[i1 - 1] if i1 else "", wa[i2] if i2 < len(wa) else "")
            out += wb[j1:j2] if keep else wa[i1:i2]
        return " ".join(out)

    # ---- all together ----
    def correct(self, original: str, humanized: str, rewrite, detect, score_doc=None, progress=lambda m: None) -> dict:
        """rewrite(passage) -> list of candidate rewrites (the humanizer); detect(texts) -> AI probabilities per
        passage; score_doc(text) -> the detector's whole-text score, used to undo changes that push it over."""
        self.grammar_fixes = []
        cleaned = unrepeat(humanized)
        op, hp = paragraphs(original), paragraphs(cleaned)
        if len(op) != len(hp):  # paragraphs don't line up: check the text as one
            op, hp = [" ".join(op)], [" ".join(hp)]
        out, notes = [], {"restored": 0, "removed": 0, "grammar": 0, "paragraphs_rewritten": 0,
                          "repeats_cut": len(humanized.split()) - len(cleaned.split())}
        for k, (o, h) in enumerate(zip(op, hp), 1):
            progress(f"meaning {k}/{len(op)}")
            c = self.check(o, h)
            if self.problems(c) and self.gentle:
                h2, n = self.repair(o, h, c, rewrite, detect)
                notes["restored"] += n["restored"]
                notes["removed"] += n["removed"]
                h = h2
            elif self.problems(c):
                patched = self.patch(c)
                new = []
                for passage in (" ".join(sentences(patched)[i:i + 4]) for i in range(0, len(sentences(patched)), 4)):
                    cands = [x for x in rewrite(passage) if x.strip()]
                    new.append(self.best(passage, o, cands, detect(cands)) if cands else passage)
                cand = " ".join(new)
                c2 = self.check(o, cand)
                was, now = detect([h, cand])
                # fewer problems, and not pushed over the line (or, if it was already over, not pushed further)
                if self.problems(c2) < self.problems(c) and now <= max(LINE, was + 0.05):
                    notes["restored"] += len(c["missing"]) - len(c2["missing"])
                    notes["removed"] += len(c["wrong"]) - len(c2["wrong"])
                    notes["paragraphs_rewritten"] += 1
                    h = cand
            out.append(h)
        progress("grammar")
        for k, h in enumerate(out):
            s = sentences(h)
            fixed = self.fix_grammar(s)
            base = detect([h])[0]
            for i, f in enumerate(fixed):
                if self.safe_check:
                    f = self.safe(s[i], f)
                if f == s[i] or not self.small_edit(s[i], f):
                    continue
                trial = s[:i] + [f] + s[i + 1:]
                p = detect([" ".join(trial)])[0]
                if p <= base + 0.03 or p < 0.15:
                    self.grammar_fixes.append((s[i], f))
                    s, base = trial, p
                    notes["grammar"] += 1
            out[k] = " ".join(s)
        if score_doc is not None:  # whole-text gate: undo the paragraphs that raised the score most until it's back
            limit = max(LINE, score_doc(cleaned))
            order = sorted(range(len(out)), key=lambda i: detect([hp[i]])[0] - detect([out[i]])[0])
            while score_doc("\n\n".join(out)) > limit and order:
                i = order.pop(0)
                if out[i] != hp[i]:
                    out[i] = hp[i]
                    notes["undone"] = notes.get("undone", 0) + 1
        text = "\n\n".join(out)
        return {"text": text, "diff": diff(humanized, text), "notes": notes}


GRAMMAR_WORDS = set("""a an the this that these those some any each every no all both either neither another such
in on at of for to with about from by into onto over under after before during since until through between among
within without against toward towards upon off out up down as than like per via around across along behind beyond
and or but nor so yet if because although though while when where whether which who whom whose what why how
is are was were be been being am has have had having do does did done will would shall should can could may might
must it its they them their there we us our you your he him his she her i me my one ones not more most much many
less few also too only just even then""".split())
PRONOUNS = {"it", "they", "them", "we", "us", "you", "he", "him", "she", "her", "i", "me", "this", "these", "those"}
SAFE_PUNCT = set(",.;?!'")


def bare(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w).lower()


def safe_change(a: list[str], b: list[str], before: str = "", after: str = "") -> bool:
    """Is changing words a into b a grammar-sized fix? Rejected: removing a pronoun, "of" or "a"/"an", or any other
    word unless it is "the", "that" or a repeat; changing a word into one that doesn't share its first 3 letters
    (unless both are small grammar words: "on" -> "of", "is" -> "are"); adding a word that isn't a small grammar
    word or a form of a removed one; adding punctuation other than , . ; ? ! and '."""
    added = set("".join(re.sub(r"[\w]", "", w) for w in b)) - set("".join(re.sub(r"[\w]", "", w) for w in a))
    if added - SAFE_PUNCT:
        return False
    ba, bb = [bare(w) for w in a if bare(w)], [bare(w) for w in b if bare(w)]
    if ba == bb:  # only punctuation or capitals changed: capitals only at the start of the sentence
        return [re.sub(r"\W", "", w) for w in a] == [re.sub(r"\W", "", w) for w in b] or not before
    related = lambda x, ys: any(x[:3] == y[:3] for y in ys)
    for x in ba:
        if x in bb or related(x, bb) or (x in GRAMMAR_WORDS and bb and all(y in GRAMMAR_WORDS for y in bb)):
            continue
        if x in PRONOUNS or x in {"of", "a", "an"}:
            return False
        if not (x in {"the", "that"} or x in (bare(before), bare(after))):
            return False
    for y in bb:
        if not (y in ba or related(y, ba) or y in GRAMMAR_WORDS):
            return False
    return True


def respace(a: str, b: str) -> str:
    """b with a's spacing wherever the two differ only in spaces (small T5 models write "Name ]." for "Name].")."""
    wa, wb, out = a.split(), b.split(), []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        same = "".join(wa[i1:i2]) == "".join(wb[j1:j2])
        out += wa[i1:i2] if tag == "equal" or same else wb[j1:j2]
    return " ".join(out)


def diff(a: str, b: str) -> list[dict]:
    """Word-level diff: [{op: eq|del|ins, text}], whitespace kept with the words."""
    ta, tb = re.findall(r"\S+\s*", a), re.findall(r"\S+\s*", b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [t.strip() for t in ta], [t.strip() for t in tb],
                                                       autojunk=False).get_opcodes():
        if tag == "equal":
            out.append({"op": "eq", "text": "".join(tb[j1:j2])})
        else:
            if i2 > i1:
                out.append({"op": "del", "text": "".join(ta[i1:i2])})
            if j2 > j1:
                out.append({"op": "ins", "text": "".join(tb[j1:j2])})
    return out
