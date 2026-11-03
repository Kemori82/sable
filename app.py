"""Writing Studio: the detector, humanizer and writer in one window (Edge WebView2).

A loading screen shows at once while the backend starts (server.py: torch, transformers); then the app opens
on the Detector tab while DeBERTa loads. Writer and humanizer models load the first time they are used.
The server listens only on 127.0.0.1 and stops when the window closes. Nothing is downloaded: every model is in
models/.

Usage:  pythonw app.py              (the shortcut runs this)
        python app.py --browser     # no window: serve on http://127.0.0.1:8090 for a browser
"""
import argparse
import ctypes
import os
import socket
import threading
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never contact the Hugging Face Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
TITLE = "Writing Studio"
ICON = ROOT / "app.ico"

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  :root { color-scheme: light dark; --bg:#f5f5f7; --fg:#1d1d1f; --muted:#6e6e73; --track:rgba(0,0,0,.08); --accent:#0071e3; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1c1c1e; --fg:#f5f5f7; --muted:#a1a1a6; --track:rgba(255,255,255,.12); --accent:#0a84ff; } }
  html,body { height:100%; margin:0; background:var(--bg); color:var(--fg); font-family:"Segoe UI Variable Display","Segoe UI",system-ui,sans-serif; }
  .c { height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:18px; }
  h1 { font-size:22px; font-weight:600; letter-spacing:-.01em; margin:0; }
  p { margin:0; color:var(--muted); font-size:14px; white-space:pre-wrap; max-width:80vw; }
  .bar { width:180px; height:4px; border-radius:2px; background:var(--track); overflow:hidden; }
  .bar i { display:block; width:40%; height:100%; border-radius:2px; background:var(--accent); animation:s 1.2s ease-in-out infinite; }
  @keyframes s { 0% { transform:translateX(-100%); } 100% { transform:translateX(250%); } }
  @media (prefers-reduced-motion: reduce) { .bar i { animation:none; width:100%; opacity:.5; } }
</style></head><body><div class="c"><h1>Writing Studio</h1><div class="bar"><i></i></div><p id="m">Starting…</p></div></body></html>"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def set_window_icon():
    """pywebview can't set the icon on Windows, so set it on the native window directly."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, TITLE)
    if not hwnd or not ICON.exists():
        return
    for size, which in ((16, 0), (32, 1)):  # ICON_SMALL, ICON_BIG
        h = user32.LoadImageW(None, str(ICON), 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
        if h:
            user32.SendMessageW(hwnd, 0x80, which, h)  # WM_SETICON


def start_backend(window) -> None:
    try:
        import server  # torch + transformers: the slow part

        port = free_port()
        server.serve(port)
        window.load_url(f"http://127.0.0.1:{port}/")
    except Exception:
        msg = traceback.format_exc().replace("\\", "\\\\").replace("`", "'")
        window.evaluate_js(f"document.getElementById('m').innerText = `Could not start:\n{msg}`")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--browser", action="store_true")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    if args.browser:
        import server

        server.serve(args.port)
        print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
        threading.Event().wait()
    import webview

    try:  # own taskbar entry and icon instead of grouping under Python
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WritingStudio.App")
    except Exception:
        pass
    window = webview.create_window(TITLE, html=LOADING_HTML, width=1180, height=820, min_size=(760, 560),
                                   background_color="#F5F5F7", text_select=True)
    window.events.shown += set_window_icon
    webview.start(start_backend, window, private_mode=False)  # private_mode off: settings are remembered


if __name__ == "__main__":
    main()
