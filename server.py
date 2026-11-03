"""Backend: serves page.html and the API on 127.0.0.1. Imported by app.py once its loading screen is up.

  GET  /              the page, with the saved settings filled in
  POST /api/detect    {text, type}                                     -> score, band, highlighted segments
  POST /api/humanize  {text, type, model: humanizer_evo|humanizer_rl}  -> streams the rewrite, passage by passage
  POST /api/write     {type, topic, opening, model: writer_r3|writer_sft, max_tokens, temperature} -> streams text
  POST /api/stop      stops the current generation
  GET  /api/status    what is loaded / loading
  POST /api/settings  {theme, humanizer, writer}                       -> saved to settings.json

The detector (DeBERTa) loads at start. A writer or humanizer loads only when first used; one is kept at a time.
"""
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import generator
from detector import Detector

ROOT = Path(__file__).resolve().parent
PAGE, SETTINGS = ROOT / "page.html", ROOT / "settings.json"
DEFAULTS = {"theme": "system", "humanizer": "humanizer_evo", "writer": "writer_r3", "tries": "8"}
CHOICES = {"theme": ("system", "light", "dark", "sable", "aurora", "midnight", "glass", "space"), "humanizer": ("humanizer_evo", "humanizer_rl"),
           "writer": ("writer_r3", "writer_sft"), "tries": ("8", "4", "1")}
WRITE_TYPES, HUMANIZE_TYPES = ("essay", "review", "creative"), ("essay", "abstract", "review", "creative")


def load_settings() -> dict:
    try:
        saved = json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        saved = {}
    return {k: saved[k] if saved.get(k) in CHOICES[k] else v for k, v in DEFAULTS.items()}


class State:
    def __init__(self):
        self.detector = None
        self.det_status, self.gen_status, self.error = "loading", "idle", None
        self.tok = self.model = self.name = None
        self.gpu = threading.Lock()  # one load / generation / detection at a time
        self.stop = threading.Event()

    def load_detector(self):
        try:
            self.detector = Detector()
            self.det_status = "ready"
        except Exception:
            self.det_status, self.error = "error", traceback.format_exc()

    def free_generator(self):
        self.model = self.tok = self.name = None
        self.gen_status = "idle"
        torch.cuda.empty_cache()

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
        elif self.path == "/api/status":
            self.json({"detector": S.det_status, "generator": S.gen_status, "model": S.name,
                       "label": generator.LABELS.get(S.name or (S.gen_status.split(" ") + [""])[1]),
                       "error": S.error})
        else:
            self.send(404, b"{}")

    def do_POST(self):
        if self.path == "/api/stop":
            S.stop.set()
            return self.json({})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            if self.path == "/api/settings":
                SETTINGS.write_text(json.dumps({**load_settings(), **{k: v for k, v in req.items() if v in CHOICES.get(k, ())}}))
                return self.json(load_settings())
            if self.path == "/api/detect":
                return self.detect(req)
            if self.path == "/api/humanize":
                return self.humanize(req)
            if self.path == "/api/write":
                return self.write(req)
        except Exception:
            return self.json({"error": traceback.format_exc()}, 500)
        self.send(404, b"{}")

    def detect(self, req):
        if S.det_status != "ready":
            return self.json({"error": "The detector is still loading." if S.det_status == "loading" else S.error}, 503)
        doc_type = req.get("type") if req.get("type") in ("essay", "abstract") else "other"
        with S.gpu:
            r = S.detector.score(req.get("text") or "", doc_type)
        self.json(r)

    def humanize(self, req):
        paras = generator.chunks(req.get("text") or "")
        if not paras:
            return self.json({"error": "Paste some text to humanize first."}, 400)
        name = req.get("model") if req.get("model") in ("humanizer_evo", "humanizer_rl") else "humanizer_evo"
        what = generator.WHAT.get(req.get("type") if req.get("type") in HUMANIZE_TYPES else "essay")
        temperature = max(0.1, min(1.5, float(req.get("temperature", 0.8))))
        tries = int(req.get("tries")) if str(req.get("tries")) in CHOICES["tries"] else int(DEFAULTS["tries"])
        with S.gpu:
            S.stop.clear()
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

    def best_rewrite(self, prompt: str, passage: str, max_new: int, temperature: float, n: int) -> str:
        """n rewrites of passage in one batch; the one the detector finds least AI-like, among those whose length
        is 0.7-1.5x the original (a much shorter or longer rewrite has usually dropped or added content)."""
        chat = S.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = S.tok([chat] * n, return_tensors="pt").to(S.model.device)
        out = S.model.generate(**enc, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new,
                               temperature=temperature, stopping_criteria=StoppingCriteriaList([StopFlag()]))
        cands = [c for c in (S.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out) if c]
        if not cands:
            return ""
        words = len(passage.split())
        ok = [c for c in cands if 0.7 <= len(c.split()) / words <= 1.5] or cands
        return min(zip(S.detector.passage_prob(ok), ok))[1]

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
    threading.Thread(target=S.load_detector, daemon=True).start()
    return server
