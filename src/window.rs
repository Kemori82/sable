//! The app window (Edge WebView2 via wry). Opens at once; downloads missing models first if needed, showing
//! progress; then starts the server and shows the page.
use crate::{install, paths, server, settings};
use std::sync::Arc;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::{WebContext, WebViewBuilder};

enum Msg {
    Progress(String),
    Ready(u16),
    Failed(String),
    Window(String), // from the page's own title bar: drag, min, max, close, resize:<edge>
}

/// Window background per theme, matching page.html (no white flash while it loads).
fn background() -> (u8, u8, u8) {
    match settings::get("theme").as_str() {
        "dark" => (0x14, 0x14, 0x16),
        "sable" => (0x12, 0x10, 0x10),
        "aurora" => (0x06, 0x17, 0x1c),
        "midnight" => (0x07, 0x0b, 0x18),
        "glass" => (0xee, 0xf0, 0xf8),
        "space" => (0x04, 0x03, 0x0a),
        "light" => (0xf5, 0xf5, 0xf7),
        _ => if system_dark() { (0x14, 0x14, 0x16) } else { (0xf5, 0xf5, 0xf7) },
    }
}

#[cfg(windows)]
fn system_dark() -> bool {
    use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_DWORD};
    let key: Vec<u16> = "Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize\0".encode_utf16().collect();
    let val: Vec<u16> = "AppsUseLightTheme\0".encode_utf16().collect();
    let mut data: u32 = 1;
    let mut size: u32 = 4;
    let rc = unsafe {
        RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), val.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(),
                     &mut data as *mut u32 as *mut _, &mut size)
    };
    rc == 0 && data == 0
}

#[cfg(not(windows))]
fn system_dark() -> bool {
    false
}

fn setup_page((r, g, b): (u8, u8, u8)) -> String {
    let dark = (r as u32 + g as u32 + b as u32) < 384;
    let (fg, muted, accent) = if dark { ("#f5f5f7", "#a1a1a6", "#0a84ff") } else { ("#1d1d1f", "#6e6e73", "#0071e3") };
    format!(r#"<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ height:100%; margin:0; background:rgb({r},{g},{b}); color:{fg}; font-family:"Segoe UI Variable Display","Segoe UI",system-ui,sans-serif; }}
  .c {{ height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:16px; padding:0 24px; text-align:center; }}
  h1 {{ font-size:24px; font-weight:600; letter-spacing:-.02em; margin:0; }}
  p {{ margin:0; color:{muted}; font-size:14px; white-space:pre-wrap; max-width:80vw; }}
  .bar {{ width:220px; height:4px; border-radius:2px; background:rgba(127,127,127,.2); overflow:hidden; }}
  .bar i {{ display:block; height:100%; width:0; background:{accent}; transition:width .3s; }}
  .bar2 {{ position:fixed; inset:0 0 auto 0; height:44px; display:flex; justify-content:flex-end; }}
  .bar2 button {{ width:46px; border:0; background:transparent; color:{fg}; font-size:16px; }}
  .bar2 button:hover {{ background:#c42b1c; color:#fff; }}
</style></head><body><div class="bar2" onmousedown="if(event.target===this)window.ipc.postMessage('drag')">
<button onclick="window.ipc.postMessage('close')" aria-label="Close">&#x2715;</button></div><div class="c"><h1>Sable</h1><p id="t">Getting the models (about 2.5 GB, once)…</p>
<div class="bar"><i id="b"></i></div><p id="m"></p></div></body></html>"#)
}

pub fn run(port: u16) {
    let event_loop = EventLoopBuilder::<Msg>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let mut wb = WindowBuilder::new()
        .with_title("Sable")
        .with_decorations(false) // the page draws the title bar itself (page.html: .winctl)
        .with_inner_size(tao::dpi::LogicalSize::new(1180.0, 820.0))
        .with_min_inner_size(tao::dpi::LogicalSize::new(760.0, 560.0));
    #[cfg(windows)]
    {
        use tao::platform::windows::IconExtWindows;
        use tao::platform::windows::WindowBuilderExtWindows;
        if let Ok(icon) = tao::window::Icon::from_resource(1, None) {
            wb = wb.with_window_icon(Some(icon));
        }
        wb = wb.with_undecorated_shadow(true);
    }
    let window = wb.build(&event_loop).expect("could not open the window");
    #[cfg(windows)]
    {
        // Windows 11's rounded corners, which a window without a system title bar doesn't get by itself
        use tao::platform::windows::WindowExtWindows;
        use windows_sys::Win32::Graphics::Dwm::{DwmSetWindowAttribute, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND};
        let pref = DWMWCP_ROUND;
        unsafe {
            DwmSetWindowAttribute(window.hwnd() as _, DWMWA_WINDOW_CORNER_PREFERENCE as u32, &pref as *const _ as *const _, 4);
        }
    }
    let bg = background();
    let data = std::env::var("LOCALAPPDATA").map(|p| std::path::PathBuf::from(p).join("Sable")).unwrap_or_else(|_| paths::app().join("webview"));
    let mut ctx = WebContext::new(Some(data));
    let needs_models = !install::complete();
    let ipc = proxy.clone();
    let builder = WebViewBuilder::new_with_web_context(&mut ctx).with_background_color((bg.0, bg.1, bg.2, 255))
        .with_ipc_handler(move |req| { let _ = ipc.send_event(Msg::Window(req.body().clone())); });
    let builder = if needs_models { builder.with_html(setup_page(bg)) } else { builder.with_html(format!("<html><body style=\"background:rgb({},{},{})\"></body></html>", bg.0, bg.1, bg.2)) };
    let webview = builder.build(&window).expect("could not start WebView2 (Microsoft Edge WebView2 Runtime is needed)");

    std::thread::spawn(move || {
        if needs_models {
            let p = proxy.clone();
            if let Err(e) = install::run(&mut |line| { let _ = p.send_event(Msg::Progress(line.to_string())); }) {
                let _ = proxy.send_event(Msg::Failed(format!("{e:#}")));
                return;
            }
        }
        let state = Arc::new(server::State::default());
        match server::serve(port, state.clone()) {
            Ok(port) => {
                state.set_detector(settings::get("detector") == "on");
                let _ = proxy.send_event(Msg::Ready(port));
            }
            Err(e) => {
                let _ = proxy.send_event(Msg::Failed(format!("Could not start: {e:#}")));
            }
        }
    });

    event_loop.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        match event {
            Event::UserEvent(Msg::Ready(port)) => {
                let _ = webview.load_url(&format!("http://127.0.0.1:{port}/"));
            }
            Event::UserEvent(Msg::Progress(line)) => {
                // "PROGRESS done total" lines move the bar; anything else is shown as text
                let js = if let Some(rest) = line.strip_prefix("PROGRESS ") {
                    let v: Vec<f64> = rest.split(' ').filter_map(|x| x.parse().ok()).collect();
                    let pct = if v.len() >= 2 && v[1] > 0.0 { 100.0 * v[0] / v[1] } else { 0.0 };
                    format!("document.getElementById('b').style.width='{pct:.1}%';document.getElementById('m').textContent={};",
                            serde_json::to_string(&format!("{:.0} / {:.0} MB", v.first().unwrap_or(&0.0) / 1048576.0, v.get(1).unwrap_or(&0.0) / 1048576.0)).unwrap())
                } else {
                    format!("document.getElementById('t').textContent={};", serde_json::to_string(&line).unwrap())
                };
                let _ = webview.evaluate_script(&js);
            }
            Event::UserEvent(Msg::Failed(msg)) => {
                let js = format!("document.getElementById('t')?(document.getElementById('t').textContent={m}):document.body.innerText={m};",
                                 m = serde_json::to_string(&msg).unwrap());
                let _ = webview.evaluate_script(&js);
            }
            Event::UserEvent(Msg::Window(m)) => {
                use tao::window::ResizeDirection as D;
                match m.as_str() {
                    "drag" => { let _ = window.drag_window(); }
                    "min" => window.set_minimized(true),
                    "max" => window.set_maximized(!window.is_maximized()),
                    "close" => std::process::exit(0),
                    r => {
                        let dir = match r.strip_prefix("resize:") {
                            Some("n") => Some(D::North), Some("s") => Some(D::South), Some("e") => Some(D::East),
                            Some("w") => Some(D::West), Some("ne") => Some(D::NorthEast), Some("nw") => Some(D::NorthWest),
                            Some("se") => Some(D::SouthEast), Some("sw") => Some(D::SouthWest), _ => None,
                        };
                        if let Some(d) = dir {
                            let _ = window.drag_resize_window(d);
                        }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                // the maximize button shows "restore" while maximized, and the resize edges go away
                let _ = webview.evaluate_script(&format!("window.__sableMax&&window.__sableMax({})", window.is_maximized()));
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *flow = ControlFlow::Exit;
                std::process::exit(0);
            }
            _ => {}
        }
    });
}
