//! Where things are: the app folder (next to Sable.exe), its models/ and settings.json.
use std::path::PathBuf;
use std::sync::OnceLock;

pub fn app() -> &'static PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        if let Ok(p) = std::env::var("SABLE_HOME") {
            return PathBuf::from(p);
        }
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| ".".into())
    })
}

pub fn models() -> PathBuf {
    std::env::var("SABLE_MODELS").map(PathBuf::from).unwrap_or_else(|_| app().join("models"))
}

pub fn settings() -> PathBuf {
    app().join("settings.json")
}
