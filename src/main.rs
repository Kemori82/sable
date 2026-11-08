//! Sable: the detector, humanizer and writer in one window (Edge WebView2), all on this computer.
//!
//! The window opens at once on the Detector tab while the detector loads; writer and humanizer models load the
//! first time they are used. The server listens only on 127.0.0.1 and stops when the window closes. If models
//! are missing (first start, or an update added some), they are downloaded first, with progress in the window.
//!
//!   Sable.exe                      the app
//!   Sable.exe --browser [--port N] no window: serve on http://127.0.0.1:N (default 8090) for a browser
//!   Sable.exe parity ...           checks against the Python version (development)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod corrector;
mod detector;
mod difflib;
mod gec;
mod generator;
mod install;
mod llm;
mod nli;
mod onnx;
mod parity;
mod paths;
mod server;
mod settings;
mod text;
mod updater;
mod window;

use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    #[cfg(windows)]
    if args.iter().any(|a| a == "parity" || a == "--browser") {
        // print to the terminal it was started from (the app itself has no console window)
        unsafe { windows_sys::Win32::System::Console::AttachConsole(windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS) };
    }
    if args.get(1).map(|a| a == "parity").unwrap_or(false) {
        if let Err(e) = parity::main(&args[2..]) {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
        return;
    }
    let browser = args.iter().any(|a| a == "--browser");
    let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok())
        .unwrap_or(if browser { 8090 } else { 0 });
    if browser {
        if !install::complete() {
            println!("Downloading models first…");
            if let Err(e) = install::run(&mut |line| println!("{line}")) {
                eprintln!("{e:#}");
                std::process::exit(1);
            }
        }
        let state = Arc::new(server::State::default());
        let port = server::serve(port, state.clone()).expect("could not start the server");
        state.set_detector(settings::get("detector") == "on");
        println!("open http://127.0.0.1:{port}  (Ctrl+C to stop)");
        loop {
            std::thread::park();
        }
    }
    updater::clean();
    window::run(port);
}
