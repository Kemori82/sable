//! Updates from GitHub releases: the latest release tagged v<version> has Sable.exe and the files next to it
//! as assets. A newer version's changed files are downloaded beside the old ones, checked against the SHA-256
//! GitHub lists for them, then swapped in (a running .exe or .dll can be renamed on Windows, not overwritten:
//! the old one becomes *.old and is deleted on the next start). New models are fetched by the new version when
//! it starts (models.json is built into Sable.exe).
use crate::{install, paths};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::time::Duration;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const APP_FILES: &[&str] = &["Sable.exe", "DirectML.dll", "vulkan-1.dll", "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"];

fn repo() -> String {
    serde_json::from_str::<Value>(install::MANIFEST).ok().and_then(|m| m["repo"].as_str().map(String::from))
        .unwrap_or_else(|| "Kemori82/sable".into())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(60))).build().into()
}

/// The newest app release (tag v<version>; model releases are tagged models-v<n> and skipped).
fn latest() -> Result<Value> {
    let url = format!("https://api.github.com/repos/{}/releases?per_page=30", repo());
    let mut r = agent().get(&url).header("User-Agent", "Sable-updater").header("Accept", "application/vnd.github+json").call()?;
    let all: Vec<Value> = serde_json::from_reader(r.body_mut().as_reader())?;
    let ver = |v: &Value| -> Vec<u64> { v["tag_name"].as_str().unwrap_or("")[1..].split('.').map(|x| x.parse().unwrap_or(0)).collect() };
    all.into_iter()
        .filter(|v| !v["draft"].as_bool().unwrap_or(false) && !v["prerelease"].as_bool().unwrap_or(false))
        .filter(|v| v["tag_name"].as_str().map(|t| t.starts_with('v') && t[1..].starts_with(|c: char| c.is_ascii_digit())).unwrap_or(false))
        .max_by_key(|v| ver(v))
        .ok_or(anyhow!("no app release yet"))
}

fn newer(tag: &str) -> bool {
    let v = |s: &str| -> Vec<u64> { s.trim_start_matches(|c: char| !c.is_ascii_digit()).split('.').map(|x| x.parse().unwrap_or(0)).collect() };
    v(tag) > v(VERSION)
}

/// The release's app files that differ from the ones here: (name, url, sha256).
fn changed(rel: &Value) -> Vec<(String, String, String)> {
    let mut out = vec![];
    for a in rel["assets"].as_array().cloned().unwrap_or_default() {
        let name = a["name"].as_str().unwrap_or("").to_string();
        if !APP_FILES.contains(&name.as_str()) {
            continue;
        }
        let digest = a["digest"].as_str().unwrap_or("").trim_start_matches("sha256:").to_string();
        let local = install::sha256(&paths::app().join(&name)).unwrap_or_default();
        if digest.is_empty() || local != digest {
            out.push((name, a["browser_download_url"].as_str().unwrap_or("").to_string(), digest));
        }
    }
    out
}

pub fn check() -> Result<Value> {
    if paths::app().join("Cargo.toml").exists() || paths::app().ancestors().any(|p| p.join(".git").exists()) {
        return Ok(json!({"available": false, "mode": "dev", "files": [],
                         "note": "This is a development copy: update it with git pull and cargo build."}));
    }
    let rel = match latest() {
        Ok(r) => r,
        Err(_) => return Ok(json!({"available": false, "files": [], "mode": "release"})),
    };
    let tag = rel["tag_name"].as_str().unwrap_or("");
    let files: Vec<String> = if newer(tag) { changed(&rel).into_iter().map(|f| f.0).collect() } else { vec![] };
    Ok(json!({"available": !files.is_empty(), "files": files, "mode": "release",
              "message": rel["name"].as_str().unwrap_or(tag)}))
}

pub fn apply(say: &mut dyn FnMut(&str)) -> Result<()> {
    let rel = latest()?;
    let files = changed(&rel);
    if files.is_empty() {
        say("Already up to date.");
        return Ok(());
    }
    let dir = paths::app();
    say("Downloading the latest version…");
    for (name, url, digest) in &files {
        let new = dir.join(format!("{name}.new"));
        let mut r = agent().get(url).header("User-Agent", "Sable-updater").call()?;
        let mut f = std::fs::File::create(&new)?;
        std::io::copy(&mut r.body_mut().as_reader(), &mut f)?;
        drop(f);
        if !digest.is_empty() && install::sha256(&new)? != *digest {
            let _ = std::fs::remove_file(&new);
            return Err(anyhow!("{name} did not download correctly; nothing was changed"));
        }
    }
    for (name, _, _) in &files {
        let (cur, new, old) = (dir.join(name), dir.join(format!("{name}.new")), dir.join(format!("{name}.old")));
        let _ = std::fs::remove_file(&old);
        if cur.exists() {
            std::fs::rename(&cur, &old)?;
        }
        std::fs::rename(&new, &cur)?;
    }
    say(&format!("Updated {} file{}.", files.len(), if files.len() == 1 { "" } else { "s" }));
    Ok(())
}

/// Start the app again the way it was started, then end this process (closing its window).
pub fn restart() {
    let exe = paths::app().join("Sable.exe");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let _ = std::process::Command::new(exe).args(args).spawn();
    std::process::exit(0);
}

/// Leftovers of the last update.
pub fn clean() {
    for f in APP_FILES {
        let _ = std::fs::remove_file(paths::app().join(format!("{f}.old")));
    }
}
