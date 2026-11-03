"""Sable: the detector, humanizer and writer in one window (Edge WebView2).

A loading screen shows at once while the backend starts (server.py: torch, transformers); then the app opens
on the Detector tab while DeBERTa loads. Writer and humanizer models load the first time they are used.
The server listens only on 127.0.0.1 and stops when the window closes. Nothing is downloaded: every model is in
models/.

Usage:  pythonw app.py              (the shortcut runs this)
        python app.py --browser     # no window: serve on http://127.0.0.1:8090 for a browser
"""
import argparse
import ctypes
import json
import os
import socket
import threading
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never contact the Hugging Face Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent
TITLE = "Sable"
ICON = ROOT / "app.ico"

THEMES = {  # loading-screen colors, matching page.html (bg, fg, muted, track, accent)
    "light": ("#f5f5f7", "#1d1d1f", "#6e6e73", "rgba(0,0,0,.08)", "#0071e3"),
    "dark": ("#141416", "#f5f5f7", "#a1a1a6", "rgba(255,255,255,.12)", "#0a84ff"),
    "sable": ("#121010", "#f3ede4", "#a69c90", "rgba(255,255,255,.1)", "#d4a259"),
    "aurora": ("#06171c", "#e6fffa", "#8fb8b0", "rgba(127,255,212,.14)", "#5ef0c8"),
    "midnight": ("#070b18", "#e8edff", "#8e9bc4", "rgba(122,162,255,.14)", "#7aa2ff"),
    "glass": ("#eef0f8", "#15161c", "#4d5263", "rgba(0,0,0,.08)", "#1f6bff"),
    "space": ("#04030a", "#f1ecff", "#a49bc2", "rgba(190,150,255,.14)", "#b48cff"),
}
LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html,body { height:100%; margin:0; background:BG; color:FG; font-family:"Segoe UI Variable Display","Segoe UI",system-ui,sans-serif; }
  .c { height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:18px; }
  h1 { font-size:24px; font-weight:600; letter-spacing:-.02em; margin:0; }
  p { margin:0; color:MUTED; font-size:14px; white-space:pre-wrap; max-width:80vw; }
  .bar { width:160px; height:3px; border-radius:2px; background:TRACK; overflow:hidden; }
  .bar i { display:block; width:40%; height:100%; border-radius:2px; background:ACCENT; animation:s 1.2s ease-in-out infinite; }
  @keyframes s { 0% { transform:translateX(-100%); } 100% { transform:translateX(250%); } }
  @media (prefers-reduced-motion: reduce) { .bar i { animation:none; width:100%; opacity:.5; } }
</style></head><body><div class="c"><h1>Sable</h1><div class="bar"><i></i></div><p id="m">Starting…</p></div></body></html>"""


def loading_html() -> tuple[str, str]:
    """The loading screen in the saved theme, and its background color."""
    try:
        theme = json.loads((ROOT / "settings.json").read_text()).get("theme", "system")
    except (OSError, ValueError):
        theme = "system"
    if theme not in THEMES:  # follow Windows' light / dark setting
        theme = "dark" if system_dark() else "light"
    html = LOADING_HTML
    for key, value in zip(("BG", "FG", "MUTED", "TRACK", "ACCENT"), THEMES[theme]):
        html = html.replace(key, value)
    return html, THEMES[theme][0]


def system_dark() -> bool:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


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
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sable.App")
    except Exception:
        pass
    html, bg = loading_html()
    window = webview.create_window(TITLE, html=html, width=1180, height=820, min_size=(760, 560),
                                   background_color=bg, text_select=True)
    window.events.shown += set_window_icon
    webview.start(start_backend, window, private_mode=False)  # private_mode off: settings are remembered


if __name__ == "__main__":
    main()
