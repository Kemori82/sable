"""Backend: serves page.html and the API on 127.0.0.1. Imported by app.py once its loading screen is up.

  GET  /              the page, with the saved settings filled in
  POST /api/detect    {text, type}                                     -> score, band, highlighted segments
  POST /api/humanize  {text, type, model: humanizer_writer|_evo|_rl|_chain, tries} -> streams the rewrite, passage by passage
  POST /api/write     {type, topic, opening, model: writer_r3|writer_sft, max_tokens, temperature} -> streams text
  POST /api/correct   {original, text, type, model}                    -> {text, diff, notes, scores}: meaning put back,
                      grammar fixed (corrector.py)
  POST /api/stop      stops the current generation
  GET  /api/status    what is loaded / loading
  POST /api/settings  {theme, humanizer, writer, tries, updates, detector} -> saved to settings.json; detector
                      "off" unloads the detector to free GPU memory, "on" loads it again
  GET  /api/update    is there a newer version on GitHub?              -> {available, files, message, mode, note}
  POST /api/update    update and restart                               -> streams progress lines, then RESTART

The detector (DeBERTa) loads at start unless it is turned off in Settings. A writer or humanizer loads only when first used; one is kept at a time.
"""
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import corrector
import devices
import generator
import updater
from detector import Detector

ROOT = Path(__file__).resolve().parent
PAGE, SETTINGS = ROOT / "page.html", ROOT / "settings.json"
DEFAULTS = {"theme": "system", "humanizer": "humanizer_writer", "writer": "writer_r3", "tries": "8", "updates": "on",
            "detector": "on"}
CHOICES = {"theme": ("system", "light", "dark", "sable", "aurora", "midnight", "glass", "space"),
           "humanizer": ("humanizer_writer", "humanizer_evo", "humanizer_rl", "humanizer_chain"),
           "writer": ("writer_r3", "writer_sft"), "tries": ("8", "4", "1"),
           "updates": ("on", "off"), "detector": ("on", "off")}
DEVICE = devices.name()  # shown when hovering over the status pill
WRITE_TYPES, HUMANIZE_TYPES = ("essay", "review", "creative"), ("essay", "abstract", "review", "creative")


SETTINGS_VERSION = 2  # 2: the writer-voice humanizer became the default, so earlier saved humanizer choices are reset once


def load_settings() -> dict:
    try:
        saved = json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        saved = {}
    if saved.get("v", 1) < SETTINGS_VERSION:
        saved.pop("humanizer", None)
    return {k: saved[k] if saved.get(k) in CHOICES[k] else v for k, v in DEFAULTS.items()}


def save_settings(changes: dict) -> dict:
    new = {**load_settings(), **{k: v for k, v in changes.items() if v in CHOICES.get(k, ())}}
    SETTINGS.write_text(json.dumps({**new, "v": SETTINGS_VERSION}))
    return new


class State:
    def __init__(self):
        self.detector = None
        self.det_status, self.gen_status, self.error = "loading", "idle", None
        self.tok = self.model = self.name = None
        self.step = None  # "2/3" while the chain humanizer runs
        self.task = None  # what the corrector is doing, while it runs
        self.corrector = None  # loaded the first time Correct is used
        self.gpu = threading.Lock()  # one load / generation / detection at a time
        self.stop = threading.Event()

    def load_detector(self):
        with self.gpu:
            if self.detector is not None:
                return
            self.det_status = "loading"
            try:
                self.detector = Detector()
                self.det_status = "ready"
            except Exception:
                self.det_status, self.error = "error", traceback.format_exc()

    def unload_detector(self):
        with self.gpu:  # waits for anything using it to finish
            self.detector = self.corrector = None  # the corrector can't run without the detector, so it goes too
            self.det_status = "off"
            devices.empty_cache()

    def set_detector(self, on: bool):
        threading.Thread(target=self.load_detector if on else self.unload_detector, daemon=True).start()

    def free_generator(self):
        self.model = self.tok = self.name = None
        self.gen_status = "idle"
        devices.empty_cache()

    def ensure_generator(self, name: str):
        if self.name == name:
            return
        self.free_generator()
        self.gen_status = f"loading {name}"
        try:
            self.tok, self.model = generator.load(name, progress=lambda m: setattr(self, "gen_status", f"loading {name} ({m})"))
            self.name, self.gen_status = name, "ready"
        except Exception:
            self.gen_status, self.error = "error", traceback.format_exc()
            raise


S = State()


class StopFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kw):
        return S.stop.is_set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/":
            page = PAGE.read_text(encoding="utf-8").replace("/*SETTINGS*/{}", json.dumps(load_settings()))
            self.send(200, page.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/update":
            try:
                self.json(updater.check())
            except Exception as e:  # offline, GitHub unreachable, no git...
                self.json({"available": False, "error": str(e)})
        elif self.path == "/api/status":
            self.json({"detector": S.det_status, "generator": S.gen_status, "model": S.name,
                       "label": generator.LABELS.get(S.name or (S.gen_status.split(" ") + [""])[1]),
                       "step": S.step, "task": S.task, "error": S.error, "device": DEVICE})
        else:
            self.send(404, b"{}")

    def do_POST(self):
        if self.path == "/api/stop":
            S.stop.set()
            return self.json({})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if self.path == "/api/settings":
                new = save_settings(req)
                if "detector" in req:
                    S.set_detector(new["detector"] == "on")
                return self.json(new)
            if self.path == "/api/detect":
                return self.detect(req)
            if self.path == "/api/update":
                return self.update()
            if self.path == "/api/humanize":
                return self.humanize(req)
            if self.path == "/api/write":
                return self.write(req)
            if self.path == "/api/correct":
                return self.correct(req)
        except Exception:
            return self.json({"error": traceback.format_exc()}, 500)
        self.send(404, b"{}")

    def detect(self, req):
        doc_type = req.get("type") if req.get("type") in ("essay", "abstract") else "other"
        if S.det_status == "ready":
            with S.gpu:
                det = S.detector  # None if it was turned off while this waited
                r = det.score(req.get("text") or "", doc_type) if det else None
            if r is not None:
                return self.json(r)
        self.json({"error": {"loading": "The detector is still loading.",
                             "off": "The detector is off. Turn it on in Settings."}.get(S.det_status, S.error)}, 503)

    def update(self):
        S.stop.set()  # finish any writing first, then nothing else runs while files change
        with S.gpu:
            self.start_stream()
            try:
                updater.apply(lambda line: self.write_piece(line + "\n"))
            except Exception as e:
                self.write_piece(f"ERROR {e}\n")
                return
            self.write_piece("RESTART\n")
        threading.Timer(0.5, updater.restart).start()

    def humanize(self, req):
        paras = generator.chunks(req.get("text") or "")
        if not paras:
            return self.json({"error": "Paste some text to humanize first."}, 400)
        name = req.get("model") if req.get("model") in CHOICES["humanizer"] else DEFAULTS["humanizer"]
        what = generator.WHAT.get(req.get("type") if req.get("type") in HUMANIZE_TYPES else "essay")
        temperature = max(0.1, min(1.5, float(req.get("temperature", 0.8))))
        tries = int(req.get("tries")) if str(req.get("tries")) in CHOICES["tries"] else int(DEFAULTS["tries"])
        with S.gpu:
            if S.detector is None:  # turned off (or still loading): nothing to pick the best try with
                tries = 1
            S.stop.clear()
            if name == "humanizer_chain":
                try:
                    return self.humanize_chain(paras, what, temperature, tries)
                finally:
                    S.step = None
            S.ensure_generator(name)
            self.start_stream()
            for i, para in enumerate(paras):
                for j, passage in enumerate(para):
                    if S.stop.is_set():
                        return
                    if i or j:
                        self.write_piece("\n\n" if j == 0 else " ")
                    max_new = int(len(S.tok(passage).input_ids) * 1.8) + 32
                    prompt = generator.HUMANIZE.format(what=what, text=passage)
                    if tries == 1:
                        if not self.stream(prompt, max_new, temperature, strip=True):
                            return
                    elif not self.write_piece(self.best_rewrite(prompt, passage, max_new, temperature, tries)):
                        return

    def humanize_chain(self, paras, what: str, temperature: float, tries: int):
        """Every passage rewritten by v2 RL, then Writer voice (one try each, in batches), then Evolved, which gets
        the detector check: best of `tries` per passage, or streamed as it is written when the check is off."""
        texts = [p for para in paras for p in para]
        steps = generator.CHAIN_STEPS
        for k, step in enumerate(steps, 1):
            S.step = f"{k}/{len(steps)}"
            S.ensure_generator(step)
            if k < len(steps):
                texts = self.rewrite_batch(texts, what, temperature)
            if S.stop.is_set():
                return self.start_stream()
        self.start_stream()
        i = 0
        for pi, para in enumerate(paras):
            for j in range(len(para)):
                if S.stop.is_set():
                    return
                if pi or j:
                    self.write_piece("\n\n" if j == 0 else " ")
                max_new = int(len(S.tok(texts[i]).input_ids) * 1.8) + 32
                prompt = generator.HUMANIZE.format(what=what, text=texts[i])
                if tries == 1:
                    if not self.stream(prompt, max_new, temperature, strip=True):
                        return
                elif not self.write_piece(self.best_rewrite(prompt, texts[i], max_new, temperature, tries)):
                    return
                i += 1

    def rewrite_batch(self, texts: list[str], what: str, temperature: float, size: int = 8) -> list[str]:
        """One rewrite of each text, size at a time; a text whose rewrite comes back empty is kept as it was."""
        S.tok.padding_side = "left"
        out = []
        for g in range(0, len(texts), size):
            group = texts[g:g + size]
            chats = [S.tok.apply_chat_template([{"role": "user", "content": generator.HUMANIZE.format(what=what, text=t)}],
                                               tokenize=False, add_generation_prompt=True) for t in group]
            enc = S.tok(chats, return_tensors="pt", padding=True).to(S.model.device)
            gen = S.model.generate(**enc, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                   max_new_tokens=max(int(len(S.tok(t).input_ids) * 1.8) + 32 for t in group),
                                   temperature=temperature, stopping_criteria=StoppingCriteriaList([StopFlag()]))
            out += [S.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() or t for t, o in zip(group, gen)]
            if S.stop.is_set():
                return out + texts[len(out):]
        return out

    def best_rewrite(self, prompt: str, passage: str, max_new: int, temperature: float, n: int) -> str:
        """n rewrites of passage in one batch; the one the detector finds least AI-like."""
        ok = self.rewrites(prompt, passage, max_new, temperature, n)
        return min(zip(S.detector.passage_prob(ok), ok))[1] if ok else ""

    def rewrites(self, prompt: str, passage: str, max_new: int, temperature: float, n: int) -> list[str]:
        """n rewrites of passage in one batch, keeping those 0.7-1.5x the passage's length if any are (a much shorter
        or longer rewrite has usually dropped or added content)."""
        chat = S.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = S.tok([chat] * n, return_tensors="pt").to(S.model.device)
        out = S.model.generate(**enc, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new,
                               temperature=temperature, stopping_criteria=StoppingCriteriaList([StopFlag()]))
        cands = [c for c in (S.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out) if c]
        words = max(1, len(passage.split()))
        return [c for c in cands if 0.7 <= len(c.split()) / words <= 1.5] or cands

    def correct(self, req):
        """Puts back meaning the humanized text lost and fixes its grammar (corrector.py), checked by the detector."""
        original, text = (req.get("original") or "").strip(), (req.get("text") or "").strip()
        if not original or not text:
            return self.json({"error": "Nothing to correct."}, 400)
        name = req.get("model") if req.get("model") in CHOICES["humanizer"] else DEFAULTS["humanizer"]
        if name == "humanizer_chain":
            name = generator.CHAIN_STEPS[-1]  # repairs in the chain's last, neatest voice
        kind = req.get("type") if req.get("type") in HUMANIZE_TYPES else "essay"
        what, doc_type = generator.WHAT[kind], kind if kind in ("essay", "abstract") else "other"

        def rewrite(passage: str) -> list[str]:
            if S.stop.is_set():
                return []
            return self.rewrites(generator.HUMANIZE.format(what=what, text=passage), passage,
                                 int(len(S.tok(passage).input_ids) * 1.8) + 32, 0.8, 8)

        with S.gpu:
            if S.detector is None:
                return self.json({"error": "Correcting needs the detector. Turn it on in Settings."}, 503)
            S.stop.clear()
            try:
                if S.corrector is None:
                    S.task = "Loading the corrector…"
                    S.corrector = corrector.Corrector()
                S.ensure_generator(name)
                r = S.corrector.correct(original, text, rewrite, S.detector.passage_prob,
                                        score_doc=lambda t: S.detector.score(t, doc_type).get("score") or 0.0,
                                        progress=lambda m: setattr(S, "task", f"Correcting: {m}…"))
                r["scores"] = [S.detector.score(t, doc_type).get("score") for t in (text, r["text"])]
            finally:
                S.task = None
        self.json(r)

    def write(self, req):
        opening = (req.get("opening") or "").strip()
        instructions = (req.get("instructions") or "").strip()[:2000] or None
        if not opening and not instructions:
            return self.json({"error": "Give it an opening sentence or say what it should write."}, 400)
        domain = req.get("type") if req.get("type") in WRITE_TYPES else "essay"
        name = req.get("model") if req.get("model") in ("writer_r3", "writer_sft") else "writer_r3"
        topic = (req.get("topic") or "").strip() or None if domain == "essay" else None
        with S.gpu:
            S.stop.clear()
            S.ensure_generator(name)
            self.start_stream()
            self.stream(generator.write_prompt(domain, opening, topic, instructions),
                        max(32, min(1024, int(req.get("max_tokens", 640)))),
                        max(0.1, min(1.5, float(req.get("temperature", 0.9)))))

    def start_stream(self):
        self.send_response(200)  # no Content-Length: streams until the connection closes
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def write_piece(self, s: str) -> bool:
        try:
            self.wfile.write(s.encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            S.stop.set()  # window closed or request aborted
            return False

    def stream(self, prompt: str, max_new: int, temperature: float, strip: bool = False) -> bool:
        chat = S.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = S.tok(chat, return_tensors="pt").to(S.model.device)
        streamer = TextIteratorStreamer(S.tok, skip_prompt=True, skip_special_tokens=True)
        # sampled as in training: the checkpoint's generation_config would otherwise add top_k=20, repetition_penalty=1.1
        kw = dict(**enc, streamer=streamer, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0,
                  max_new_tokens=max_new, temperature=temperature, stopping_criteria=StoppingCriteriaList([StopFlag()]))
        worker = threading.Thread(target=lambda: S.model.generate(**kw), daemon=True)
        worker.start()
        ok, first = True, strip
        for piece in streamer:
            if first:  # no leading space at the start of a humanized passage
                piece = piece.lstrip()
                first = not piece
            if ok and piece:
                ok = self.write_piece(piece)
        worker.join()
        return ok


def serve(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    S.set_detector(load_settings()["detector"] == "on")
    return server
