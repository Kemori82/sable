//! settings.json: the same keys, choices and defaults as the Python version, so saved settings carry over.
use crate::paths;
use serde_json::{json, Map, Value};

pub const DEFAULTS: &[(&str, &str)] = &[("theme", "system"), ("humanizer", "humanizer_writer"), ("writer", "writer_r3"),
    ("tries", "8"), ("updates", "on"), ("detector", "on"), ("grammar", "gec_standard")];

pub fn choices(key: &str) -> &'static [&'static str] {
    match key {
        "theme" => &["system", "light", "dark", "sable", "aurora", "midnight", "glass", "space"],
        "humanizer" => &["humanizer_writer", "humanizer_evo", "humanizer_rl", "humanizer_chain"],
        "writer" => &["writer_r3", "writer_sft"],
        "tries" => &["8", "4", "1"],
        "updates" | "detector" => &["on", "off"],
        "grammar" => &["gec_standard", "gec_tuned"],
        _ => &[],
    }
}

const VERSION: i64 = 2; // 2: the writer-voice humanizer became the default, so earlier saved choices are reset once

pub fn default(key: &str) -> &'static str {
    DEFAULTS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v).unwrap_or("")
}

pub fn load() -> Map<String, Value> {
    let mut saved: Map<String, Value> = std::fs::read_to_string(paths::settings()).ok()
        .and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    if saved.get("v").and_then(|v| v.as_i64()).unwrap_or(1) < VERSION {
        saved.remove("humanizer");
    }
    DEFAULTS.iter().map(|(k, d)| {
        let v = saved.get(*k).and_then(|v| v.as_str()).filter(|v| choices(k).contains(v)).unwrap_or(d);
        (k.to_string(), json!(v))
    }).collect()
}

pub fn get(key: &str) -> String {
    load().get(key).and_then(|v| v.as_str()).unwrap_or(default(key)).to_string()
}

pub fn save(changes: &Map<String, Value>) -> Map<String, Value> {
    let mut new = load();
    for (k, v) in changes {
        if let Some(s) = v.as_str() {
            if choices(k).contains(&s) {
                new.insert(k.clone(), json!(s));
            }
        }
    }
    let mut file = new.clone();
    file.insert("v".into(), json!(VERSION));
    let _ = std::fs::write(paths::settings(), serde_json::to_string(&file).unwrap());
    new
}
