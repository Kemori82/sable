//! Downloads every model file listed in models.json (built into Sable.exe) into models/, checking each one's
//! SHA-256, as install_models.py did: 32 MB pieces over 8 connections, a piece that stalls or runs far slower
//! than the others is fetched again, files already there are skipped, and an interrupted download resumes.
//! A file can come in several release assets (GitHub's limit is 2 GB each); they are joined in order.
use crate::paths;
use anyhow::{anyhow, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const MANIFEST: &str = include_str!("../models.json");
const PIECE: u64 = 32 << 20;
const WORKERS: usize = 8;
const TRIES: u32 = 6;

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("models.json")
}

/// Every file there with the right size, and models/.complete written for this manifest.
pub fn complete() -> bool {
    let m = manifest();
    let dir = paths::models();
    let tag = m["tag"].as_str().unwrap_or("");
    if std::fs::read_to_string(dir.join(".complete")).map(|t| t.trim() != tag).unwrap_or(true) {
        return false;
    }
    m["files"].as_array().unwrap().iter().all(|e| {
        std::fs::metadata(dir.join(e["path"].as_str().unwrap())).map(|md| md.len() == e["size"].as_u64().unwrap()).unwrap_or(false)
    })
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn asset_url(m: &Value, asset: &str) -> String {
    format!("https://github.com/{}/releases/download/{}/{}", m["repo"].as_str().unwrap(), m["tag"].as_str().unwrap(), asset)
}

struct Progress {
    done: AtomicI64,
    total: i64,
    started: Instant,
    say: Mutex<Box<dyn FnMut(&str) + Send>>,
    last: Mutex<Instant>,
}

impl Progress {
    fn add(&self, n: i64) {
        let d = self.done.fetch_add(n, Ordering::Relaxed) + n;
        let mut last = self.last.lock().unwrap();
        if last.elapsed() > Duration::from_millis(300) || d >= self.total {
            *last = Instant::now();
            (self.say.lock().unwrap())(&format!("PROGRESS {d} {}", self.total));
        }
    }

    /// Bytes per second per connection, over the whole download so far.
    fn per_connection(&self) -> f64 {
        self.done.load(Ordering::Relaxed) as f64 / self.started.elapsed().as_secs_f64().max(1.0) / WORKERS as f64
    }
}

/// Bytes [start, end) of url into part at the same offset, retrying slow or failed attempts.
fn fetch_piece(url: &str, part: &Path, start: u64, end: u64, whole: bool, p: &Progress) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_recv_body(Some(Duration::from_secs(20)))
        .timeout_connect(Some(Duration::from_secs(20))).build().into();
    let mut last_err = anyhow!("not tried");
    for attempt in 1..=TRIES {
        let mut got = 0u64;
        let r = (|| -> Result<()> {
            let mut req = agent.get(url).header("User-Agent", "Sable-installer");
            if !whole {
                req = req.header("Range", &format!("bytes={}-{}", start, end - 1));
            }
            let mut resp = req.call()?;
            if !whole && resp.status().as_u16() != 206 {
                return Err(anyhow!("server sent status {} instead of a range", resp.status()));
            }
            let mut body = resp.body_mut().as_reader();
            let mut f = OpenOptions::new().write(true).open(part)?;
            f.seek(SeekFrom::Start(start))?;
            let t0 = Instant::now();
            let mut buf = vec![0u8; 1 << 20];
            while got < end - start {
                let want = buf.len().min((end - start - got) as usize);
                let n = body.read(&mut buf[..want])?;
                if n == 0 {
                    return Err(anyhow!("connection closed early"));
                }
                f.write_all(&buf[..n])?;
                got += n as u64;
                p.add(n as i64);
                let el = t0.elapsed().as_secs_f64();
                if el > 15.0 && (got as f64 / el) < 0.25 * p.per_connection() {
                    return Err(anyhow!("much slower than the other connections"));
                }
            }
            Ok(())
        })();
        match r {
            Ok(()) => return Ok(()),
            Err(e) => {
                p.add(-(got as i64));
                last_err = e;
                std::thread::sleep(Duration::from_secs(attempt as u64));
            }
        }
    }
    Err(last_err)
}

/// One asset into dst, in pieces, resuming from the pieces already there.
fn download(url: &str, size: u64, dst: &Path, p: &Arc<Progress>, pool: &Pool) -> Result<()> {
    let part = with_suffix(dst, ".part");
    let done_file = with_suffix(dst, ".part.done");
    let mut done: BTreeSet<u64> = if part.exists() && done_file.exists() {
        serde_json::from_str(&std::fs::read_to_string(&done_file)?).unwrap_or_default()
    } else {
        BTreeSet::new()
    };
    if std::fs::metadata(&part).map(|m| m.len() != size).unwrap_or(true) {
        File::create(&part)?.set_len(size)?;
        done.clear();
    }
    let starts: Vec<u64> = if size == 0 { vec![0] } else { (0..size).step_by(PIECE as usize).collect() };
    p.add(starts.iter().filter(|s| done.contains(s)).map(|s| (PIECE.min(size - s)) as i64).sum());
    let whole = starts.len() == 1;
    let done = Arc::new(Mutex::new(done));
    let todo: Vec<u64> = starts.iter().copied().filter(|s| !done.lock().unwrap().contains(s)).collect();
    let results = pool.map(todo, {
        let (url, part, done, done_file, p) = (url.to_string(), part.clone(), done.clone(), done_file.clone(), p.clone());
        move |s| -> Result<()> {
            fetch_piece(&url, &part, s, (s + PIECE).min(size), whole, &p)?;
            let mut d = done.lock().unwrap();
            d.insert(s);
            std::fs::write(&done_file, serde_json::to_string(&*d)?)?;
            Ok(())
        }
    });
    for r in results {
        r?;
    }
    std::fs::rename(&part, dst)?;
    let _ = std::fs::remove_file(&done_file);
    Ok(())
}

fn with_suffix(p: &Path, s: &str) -> PathBuf {
    let mut n = p.as_os_str().to_owned();
    n.push(s);
    PathBuf::from(n)
}

/// A small fixed pool of threads that runs a closure over items and returns the results in order.
struct Pool;

impl Pool {
    fn map<T: Send + 'static, R: Send + 'static>(&self, items: Vec<T>, f: impl Fn(T) -> R + Send + Sync + 'static) -> Vec<R> {
        let f = Arc::new(f);
        let items = Arc::new(Mutex::new(items.into_iter().enumerate().collect::<Vec<_>>()));
        let results = Arc::new(Mutex::new(vec![]));
        let handles: Vec<_> = (0..WORKERS).map(|_| {
            let (f, items, results) = (f.clone(), items.clone(), results.clone());
            std::thread::spawn(move || loop {
                let next = items.lock().unwrap().pop();
                let Some((i, t)) = next else { break };
                let r = f(t);
                results.lock().unwrap().push((i, r));
            })
        }).collect();
        for h in handles {
            let _ = h.join();
        }
        let mut r = Arc::try_unwrap(results).ok().unwrap().into_inner().unwrap();
        r.sort_by_key(|(i, _)| *i);
        r.into_iter().map(|(_, r)| r).collect()
    }
}

pub fn run(say: &mut (dyn FnMut(&str) + Send)) -> Result<()> {
    let m = manifest();
    let dir = paths::models();
    let files = m["files"].as_array().unwrap();
    say("Checking the models…");
    let mut todo = vec![];
    for e in files {
        let dst = dir.join(e["path"].as_str().unwrap());
        let ok = std::fs::metadata(&dst).map(|md| md.len() == e["size"].as_u64().unwrap()).unwrap_or(false)
            && sha256(&dst).map(|h| h == e["sha256"].as_str().unwrap()).unwrap_or(false);
        if !ok {
            todo.push(e.clone());
        }
    }
    let total: u64 = todo.iter().map(|e| e["size"].as_u64().unwrap()).sum();
    say(&format!("Downloading the models ({:.1} GB)…", total as f64 / (1u64 << 30) as f64));
    // the callback is moved into the progress counter for the download, then the rest goes straight to say
    let lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let p = Arc::new(Progress {
        done: AtomicI64::new(0),
        total: total as i64,
        started: Instant::now(),
        say: Mutex::new(Box::new({
            let lines = lines.clone();
            move |l: &str| lines.lock().unwrap().push(l.to_string())
        })),
        last: Mutex::new(Instant::now() - Duration::from_secs(1)),
    });
    let mut failed = vec![];
    let worker = {
        let (todo, dir, m, p) = (todo.clone(), dir.clone(), m.clone(), p.clone());
        std::thread::spawn(move || {
            let mut failed = vec![];
            for e in todo {
                let path = e["path"].as_str().unwrap().to_string();
                let dst = dir.join(&path);
                let r = (|| -> Result<()> {
                    std::fs::create_dir_all(dst.parent().unwrap())?;
                    let assets: Vec<String> = match e.get("assets") {
                        Some(a) => a.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect(),
                        None => vec![e["asset"].as_str().unwrap().to_string()],
                    };
                    let sizes: Vec<u64> = match e.get("asset_sizes") {
                        Some(a) => a.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect(),
                        None => vec![e["size"].as_u64().unwrap()],
                    };
                    if assets.len() == 1 {
                        download(&asset_url(&m, &assets[0]), sizes[0], &dst, &p, &Pool)?;
                    } else {
                        let mut parts = vec![];
                        for (i, (a, s)) in assets.iter().zip(&sizes).enumerate() {
                            let pp = with_suffix(&dst, &format!(".{i}"));
                            if std::fs::metadata(&pp).map(|md| md.len() != *s).unwrap_or(true) {
                                download(&asset_url(&m, a), *s, &pp, &p, &Pool)?;
                            }
                            parts.push(pp);
                        }
                        let tmp = with_suffix(&dst, ".joining");
                        let mut out = File::create(&tmp)?;
                        for pp in &parts {
                            std::io::copy(&mut File::open(pp)?, &mut out)?;
                        }
                        drop(out);
                        std::fs::rename(&tmp, &dst)?;
                        for pp in parts {
                            let _ = std::fs::remove_file(pp);
                        }
                    }
                    if sha256(&dst)? != e["sha256"].as_str().unwrap() {
                        let _ = std::fs::remove_file(&dst);
                        return Err(anyhow!("checksum did not match"));
                    }
                    Ok(())
                })();
                if let Err(err) = r {
                    failed.push(format!("{path}: {err:#}"));
                }
            }
            failed
        })
    };
    while !worker.is_finished() {
        std::thread::sleep(Duration::from_millis(200));
        for l in lines.lock().unwrap().drain(..) {
            say(&l);
        }
    }
    failed.extend(worker.join().unwrap_or_default());
    for l in lines.lock().unwrap().drain(..) {
        say(&l);
    }
    if !failed.is_empty() {
        return Err(anyhow!("Could not download:\n  {}\nCheck the internet connection and start Sable again; it keeps what it already has.",
                           failed.join("\n  ")));
    }
    std::fs::write(dir.join(".complete"), m["tag"].as_str().unwrap())?;
    say("All models downloaded and checked.");
    shortcut();
    Ok(())
}

/// A Desktop shortcut to Sable.exe, the first time the models are set up (as the old installer made).
fn shortcut() {
    let Ok(exe) = std::env::current_exe() else { return };
    let Some(desktop) = std::env::var_os("USERPROFILE").map(|p| PathBuf::from(p).join("Desktop")) else { return };
    let lnk = desktop.join("Sable.lnk");
    if lnk.exists() || exe.to_string_lossy().contains("\\target\\") {
        return;
    }
    let q = |p: &Path| p.to_string_lossy().replace('\'', "''");
    let script = format!("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{}'); $s.TargetPath = '{}'; \
                          $s.WorkingDirectory = '{}'; $s.IconLocation = '{},0'; $s.Save()",
                         q(&lnk), q(&exe), q(exe.parent().unwrap()), q(&exe));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("powershell").args(["-NoProfile", "-Command", &script])
            .creation_flags(0x08000000).status(); // CREATE_NO_WINDOW
    }
}
