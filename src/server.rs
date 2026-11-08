//! Backend: serves the page and the API on 127.0.0.1, with the same endpoints as the Python version.
//!
//!   GET  /              the page, with the saved settings filled in
//!   POST /api/detect    {text, type}                                  -> score, band, highlighted segments
//!   POST /api/humanize  {text, type, model, tries, temperature}       -> streams the rewrite, passage by passage
//!   POST /api/write     {type, topic, opening, instructions, model, max_tokens, temperature} -> streams text
//!   POST /api/correct   {original, text, type, model}                 -> {text, diff, notes, scores}
//!   POST /api/stop      stops the current generation
//!   GET  /api/status    what is loaded / loading
//!   POST /api/settings  saved to settings.json; detector "off" unloads the detector to free GPU memory
//!   GET  /api/update    is there a newer version?   POST /api/update  update and restart (streams progress)
use crate::corrector::{Corrector, Tools};
use crate::{detector::Detector, gec::Gec, generator, llm::Llm, nli::Nli, onnx, paths, settings, text, updater};
use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const PAGE: &str = include_str!("../page.html");
const WRITE_TYPES: [&str; 3] = ["essay", "review", "creative"];
const HUMANIZE_TYPES: [&str; 4] = ["essay", "abstract", "review", "creative"];

#[derive(Default)]
pub struct State {
    detector: Mutex<Option<Arc<Detector>>>,
    det_status: Mutex<String>,
    gen_status: Mutex<String>,
    error: Mutex<Option<String>>,
    llm: Mutex<Option<Llm>>,
    name: Mutex<Option<String>>,
    step: Mutex<Option<String>>,
    task: Mutex<Option<String>>,
    corrector: Mutex<Option<Corrector>>,
    gpu: Mutex<()>, // one load / generation / detection at a time
    pub stop: AtomicBool,
    device: Mutex<String>,
}

fn set(m: &Mutex<String>, v: &str) {
    *m.lock().unwrap() = v.to_string();
}

fn get(m: &Mutex<String>) -> String {
    m.lock().unwrap().clone()
}

impl State {
    fn load_detector(self: &Arc<Self>) {
        let _g = self.gpu.lock().unwrap();
        if self.detector.lock().unwrap().is_some() {
            return;
        }
        set(&self.det_status, "loading");
        match Detector::load() {
            Ok(d) => {
                *self.detector.lock().unwrap() = Some(Arc::new(d));
                set(&self.det_status, "ready");
                self.update_device();
            }
            Err(e) => {
                set(&self.det_status, "error");
                *self.error.lock().unwrap() = Some(format!("{e:#}"));
            }
        }
    }

    fn unload_detector(self: &Arc<Self>) {
        let _g = self.gpu.lock().unwrap(); // waits for anything using it to finish
        *self.detector.lock().unwrap() = None;
        *self.corrector.lock().unwrap() = None; // the corrector can't run without the detector
        set(&self.det_status, "off");
    }

    pub fn set_detector(self: &Arc<Self>, on: bool) {
        let s = self.clone();
        std::thread::spawn(move || if on { s.load_detector() } else { s.unload_detector() });
    }

    fn update_device(&self) {
        let gpu = crate::llm::gpu_name();
        let enc = if onnx::on_gpu() { "graphics card" } else { "CPU" };
        set(&self.device, &match gpu {
            Some(g) => format!("{g} (writing: Vulkan; detector and corrector: {enc})"),
            None => format!("CPU (detector and corrector: {enc})"),
        });
    }

    fn ensure_generator(&self, name: &str) -> Result<()> {
        if self.name.lock().unwrap().as_deref() == Some(name) {
            return Ok(());
        }
        let chain = generator::chain(name).ok_or(anyhow!("unknown model {name}"))?;
        set(&self.gen_status, &format!("loading {name}"));
        let progress = |m: &str| set(&self.gen_status, &format!("loading {name} ({m})"));
        let mut llm = self.llm.lock().unwrap();
        let r = (|| -> Result<()> {
            if llm.is_none() {
                *llm = Some(Llm::load(&progress)?);
            }
            llm.as_mut().unwrap().use_chain(&chain, &progress)
        })();
        match r {
            Ok(()) => {
                *self.name.lock().unwrap() = Some(name.to_string());
                set(&self.gen_status, "ready");
                drop(llm);
                self.update_device();
                Ok(())
            }
            Err(e) => {
                *self.name.lock().unwrap() = None;
                set(&self.gen_status, "error");
                *self.error.lock().unwrap() = Some(format!("{e:#}"));
                Err(e)
            }
        }
    }

    fn detector(&self) -> Option<Arc<Detector>> {
        self.detector.lock().unwrap().clone()
    }
}

// ---- a minimal HTTP/1.0 server (one thread per request, connection closed after each response) ----

struct Req {
    method: String,
    path: String,
    body: Value,
}

struct Conn {
    stream: TcpStream,
    state: Arc<State>,
}

impl Conn {
    fn send(&mut self, code: u16, body: &[u8], ctype: &str) {
        let reason = match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", 503 => "Service Unavailable", _ => "Error" };
        let head = format!("HTTP/1.0 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n\r\n", body.len());
        let _ = self.stream.write_all(head.as_bytes()).and_then(|_| self.stream.write_all(body));
    }

    fn json(&mut self, v: Value, code: u16) {
        self.send(code, v.to_string().as_bytes(), "application/json");
    }

    fn start_stream(&mut self) {
        let _ = self.stream.write_all(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nCache-Control: no-store\r\n\r\n");
    }

    /// false (and the stop flag set) if the window closed or the request was aborted
    fn piece(&mut self, s: &str) -> bool {
        if self.stream.write_all(s.as_bytes()).and_then(|_| self.stream.flush()).is_err() {
            self.state.stop.store(true, Ordering::Relaxed);
            return false;
        }
        true
    }
}

fn read_req(stream: &mut TcpStream) -> Result<Req> {
    let mut r = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("/").to_string());
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; len.min(64 << 20)];
    r.read_exact(&mut body)?;
    let body = if body.is_empty() { json!({}) } else { serde_json::from_slice(&body).unwrap_or(json!({})) };
    Ok(Req { method, path, body })
}

pub fn serve(port: u16, state: Arc<State>) -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let port = listener.local_addr()?.port();
    state.update_device();
    set(&state.gen_status, "idle");
    set(&state.det_status, "loading");
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let state = state.clone();
            std::thread::spawn(move || {
                let mut stream = stream;
                if let Ok(req) = read_req(&mut stream) {
                    let mut c = Conn { stream, state };
                    route(&mut c, req);
                }
            });
        }
    });
    Ok(port)
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("")
}

fn num(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64().or_else(|| x.as_str().and_then(|s| s.parse().ok()))).unwrap_or(d)
}

fn route(c: &mut Conn, req: Req) {
    let st = c.state.clone();
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => {
            let page = PAGE.replace("/*SETTINGS*/{}", &Value::Object(settings::load()).to_string());
            c.send(200, page.as_bytes(), "text/html; charset=utf-8");
        }
        ("GET", "/api/status") => {
            let name = st.name.lock().unwrap().clone();
            let gen = get(&st.gen_status);
            let key = name.clone().unwrap_or_else(|| gen.split(' ').nth(1).unwrap_or("").to_string());
            c.json(json!({"detector": get(&st.det_status), "generator": gen, "model": name,
                          "label": generator::label(&key), "step": *st.step.lock().unwrap(),
                          "task": *st.task.lock().unwrap(), "error": *st.error.lock().unwrap(),
                          "device": get(&st.device)}), 200);
        }
        ("GET", "/api/update") => {
            let v = updater::check().unwrap_or_else(|e| json!({"available": false, "error": format!("{e:#}")}));
            c.json(v, 200);
        }
        ("POST", "/api/stop") => {
            st.stop.store(true, Ordering::Relaxed);
            c.json(json!({}), 200);
        }
        ("POST", path) => {
            let r = match path {
                "/api/settings" => {
                    let changes: Map<String, Value> = req.body.as_object().cloned().unwrap_or_default();
                    let new = settings::save(&changes);
                    if changes.contains_key("detector") {
                        st.set_detector(new["detector"] == "on");
                    }
                    c.json(Value::Object(new), 200);
                    Ok(())
                }
                "/api/detect" => detect(c, &req.body),
                "/api/humanize" => humanize(c, &req.body),
                "/api/write" => write(c, &req.body),
                "/api/correct" => correct(c, &req.body),
                "/api/update" => update(c),
                _ => {
                    c.send(404, b"{}", "application/json");
                    Ok(())
                }
            };
            if let Err(e) = r {
                c.json(json!({"error": format!("{e:#}")}), 500);
            }
        }
        _ => c.send(404, b"{}", "application/json"),
    }
}

fn detect(c: &mut Conn, req: &Value) -> Result<()> {
    let t = s(req, "type");
    let doc = if t == "essay" || t == "abstract" { t } else { "other" };
    let st = c.state.clone();
    if get(&st.det_status) == "ready" {
        let r = {
            let _g = st.gpu.lock().unwrap();
            st.detector().map(|d| d.score(s(req, "text"), doc)) // None if it was turned off while this waited
        };
        if let Some(r) = r {
            c.json(r?, 200);
            return Ok(());
        }
    }
    let msg = match get(&st.det_status).as_str() {
        "loading" => json!("The detector is still loading."),
        "off" => json!("The detector is off. Turn it on in Settings."),
        _ => json!(*st.error.lock().unwrap()),
    };
    c.json(json!({"error": msg}), 503);
    Ok(())
}

fn max_new(llm: &Llm, passage: &str) -> usize {
    (llm.n_tokens(passage) as f64 * 1.8) as usize + 32
}

/// n rewrites of passage in one batch, keeping those 0.7-1.5x the passage's length if any are (a much shorter
/// or longer rewrite has usually dropped or added content).
fn rewrites(st: &State, prompt: &str, passage: &str, max: usize, temperature: f32, n: usize) -> Result<Vec<String>> {
    let mut g = st.llm.lock().unwrap();
    let llm = g.as_mut().ok_or(anyhow!("no model loaded"))?;
    let chat = Llm::chat(prompt);
    let outs = llm.generate(&vec![chat; n], &vec![max; n], temperature, &st.stop, None)?;
    let cands: Vec<String> = outs.iter().map(|o| text::py_strip(o).to_string()).filter(|o| !o.is_empty()).collect();
    let words = text::n_words(passage).max(1) as f64;
    let ok: Vec<String> = cands.iter().filter(|c| (0.7..=1.5).contains(&(text::n_words(c) as f64 / words))).cloned().collect();
    Ok(if ok.is_empty() { cands } else { ok })
}

/// The one of n rewrites the detector finds least AI-like.
fn best_rewrite(st: &State, prompt: &str, passage: &str, max: usize, temperature: f32, n: usize) -> Result<String> {
    let ok = rewrites(st, prompt, passage, max, temperature, n)?;
    if ok.is_empty() {
        return Ok(String::new());
    }
    let det = st.detector().ok_or(anyhow!("the detector is off"))?;
    let p = det.passage_prob(&ok)?;
    let mut pairs: Vec<(f64, &String)> = p.into_iter().zip(&ok).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(b.1)));
    Ok(pairs[0].1.clone())
}

/// Streams one reply; no leading space at the start of a humanized passage.
fn stream(c: &mut Conn, prompt: &str, max: usize, temperature: f32, strip: bool) -> Result<bool> {
    let st = c.state.clone();
    let mut g = st.llm.lock().unwrap();
    let llm = g.as_mut().ok_or(anyhow!("no model loaded"))?;
    let (mut ok, mut first) = (true, strip);
    let mut on = |piece: &str| -> bool {
        let mut p = piece;
        if first {
            p = p.trim_start();
            first = p.is_empty();
        }
        if ok && !p.is_empty() {
            ok = c.piece(p);
        }
        ok
    };
    llm.generate(&[Llm::chat(prompt)], &[max], temperature, &st.stop, Some(&mut on))?;
    Ok(ok)
}

fn humanize(c: &mut Conn, req: &Value) -> Result<()> {
    let paras = text::chunks(s(req, "text"), 4);
    if paras.is_empty() {
        c.json(json!({"error": "Paste some text to humanize first."}), 400);
        return Ok(());
    }
    let model = s(req, "model");
    let name = if settings::choices("humanizer").contains(&model) { model } else { settings::default("humanizer") }.to_string();
    let kind = s(req, "type");
    let what = generator::what(if HUMANIZE_TYPES.contains(&kind) { kind } else { "essay" });
    let temperature = num(req, "temperature", 0.8).clamp(0.1, 1.5) as f32;
    let t = req.get("tries").map(|v| v.to_string().trim_matches('"').to_string()).unwrap_or_default();
    let mut tries: usize = if settings::choices("tries").contains(&t.as_str()) { t.parse().unwrap() } else { 8 };
    let st = c.state.clone();
    let _g = st.gpu.lock().unwrap();
    if st.detector().is_none() {
        tries = 1; // turned off (or still loading): nothing to pick the best try with
    }
    st.stop.store(false, Ordering::Relaxed);
    if name == "humanizer_chain" {
        let r = humanize_chain(c, &paras, what, temperature, tries);
        *st.step.lock().unwrap() = None;
        return r;
    }
    st.ensure_generator(&name)?;
    c.start_stream();
    let mut texts = vec![];
    for (i, para) in paras.iter().enumerate() {
        for (j, p) in para.iter().enumerate() {
            texts.push((i, j, p.clone()));
        }
    }
    write_passages(c, &texts, what, temperature, tries)
}

/// Each passage rewritten (streamed when tries is 1, else the best of tries), joined as the paragraphs were.
fn write_passages(c: &mut Conn, texts: &[(usize, usize, String)], what: &str, temperature: f32, tries: usize) -> Result<()> {
    let st = c.state.clone();
    for (i, j, passage) in texts {
        if st.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        if *i > 0 || *j > 0 {
            if !c.piece(if *j == 0 { "\n\n" } else { " " }) {
                return Ok(());
            }
        }
        let max = max_new(st.llm.lock().unwrap().as_ref().unwrap(), passage);
        let prompt = generator::humanize_prompt(what, passage);
        if tries == 1 {
            if !stream(c, &prompt, max, temperature, true)? {
                return Ok(());
            }
        } else {
            let best = best_rewrite(&st, &prompt, passage, max, temperature, tries)?;
            if !c.piece(&best) {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Every passage rewritten by v2 RL, then Writer voice (one try each, in batches), then Evolved, which gets the
/// detector check.
fn humanize_chain(c: &mut Conn, paras: &[Vec<String>], what: &str, temperature: f32, tries: usize) -> Result<()> {
    let st = c.state.clone();
    let mut texts: Vec<String> = paras.iter().flatten().cloned().collect();
    let steps = generator::CHAIN_STEPS;
    for (k, step) in steps.iter().enumerate() {
        *st.step.lock().unwrap() = Some(format!("{}/{}", k + 1, steps.len()));
        st.ensure_generator(step)?;
        if k + 1 < steps.len() {
            texts = rewrite_batch(&st, &texts, what, temperature)?;
        }
        if st.stop.load(Ordering::Relaxed) {
            c.start_stream();
            return Ok(());
        }
    }
    c.start_stream();
    let mut flat = vec![];
    let mut it = texts.into_iter();
    for (i, para) in paras.iter().enumerate() {
        for j in 0..para.len() {
            flat.push((i, j, it.next().unwrap()));
        }
    }
    write_passages(c, &flat, what, temperature, tries)
}

/// One rewrite of each text, 8 at a time; a text whose rewrite comes back empty is kept as it was.
fn rewrite_batch(st: &State, texts: &[String], what: &str, temperature: f32) -> Result<Vec<String>> {
    let mut g = st.llm.lock().unwrap();
    let llm = g.as_mut().unwrap();
    let mut out = vec![];
    for group in texts.chunks(8) {
        let prompts: Vec<String> = group.iter().map(|t| Llm::chat(&generator::humanize_prompt(what, t))).collect();
        let max = group.iter().map(|t| max_new(llm, t)).max().unwrap();
        let gen = llm.generate(&prompts, &vec![max; group.len()], temperature, &st.stop, None)?;
        for (t, o) in group.iter().zip(gen) {
            let o = text::py_strip(&o);
            out.push(if o.is_empty() { t.clone() } else { o.to_string() });
        }
        if st.stop.load(Ordering::Relaxed) {
            out.extend_from_slice(&texts[out.len()..]);
            return Ok(out);
        }
    }
    Ok(out)
}

fn correct(c: &mut Conn, req: &Value) -> Result<()> {
    let (original, humanized) = (text::py_strip(s(req, "original")).to_string(), text::py_strip(s(req, "text")).to_string());
    if original.is_empty() || humanized.is_empty() {
        c.json(json!({"error": "Nothing to correct."}), 400);
        return Ok(());
    }
    let model = s(req, "model");
    let mut name = if settings::choices("humanizer").contains(&model) { model } else { settings::default("humanizer") };
    if name == "humanizer_chain" {
        name = generator::CHAIN_STEPS[2]; // repairs in the chain's last, neatest voice
    }
    let kind = if HUMANIZE_TYPES.contains(&s(req, "type")) { s(req, "type") } else { "essay" };
    let what = generator::what(kind);
    let doc_type = if kind == "essay" || kind == "abstract" { kind } else { "other" };
    let st = c.state.clone();
    let _g = st.gpu.lock().unwrap();
    let Some(det) = st.detector() else {
        c.json(json!({"error": "Correcting needs the detector. Turn it on in Settings."}), 503);
        return Ok(());
    };
    st.stop.store(false, Ordering::Relaxed);
    let r = (|| -> Result<Value> {
        let tuned = settings::get("grammar") == "gec_tuned" && paths::models().join("gec-tuned").join("decoder.onnx").exists();
        {
            let mut cor = st.corrector.lock().unwrap();
            let reload_gec = cor.as_ref().map(|c| c.safe_check != tuned).unwrap_or(true);
            if reload_gec {
                *st.task.lock().unwrap() = Some("Loading the corrector…".into());
                let nli = match cor.take() {
                    Some(old) => old.nli,
                    None => Nli::load()?,
                };
                *cor = Some(Corrector { nli, gec: Gec::load(if tuned { "gec-tuned" } else { "gec" })?, safe_check: tuned });
            }
        }
        st.ensure_generator(name)?;
        let cor = st.corrector.lock().unwrap();
        let cor = cor.as_ref().unwrap();
        let st2 = st.clone();
        let mut rewrite = |passage: &str| -> Result<Vec<String>> {
            if st2.stop.load(Ordering::Relaxed) {
                return Ok(vec![]);
            }
            let max = max_new(st2.llm.lock().unwrap().as_ref().unwrap(), passage);
            rewrites(&st2, &generator::humanize_prompt(what, passage), passage, max, 0.8, 8)
        };
        let detect = |t: &[String]| det.passage_prob(t);
        let score_doc = |t: &str| -> Result<f64> { Ok(det.score(t, doc_type)?["score"].as_f64().unwrap_or(0.0)) };
        let progress = |m: &str| *st.task.lock().unwrap() = Some(format!("Correcting: {m}…"));
        let mut r = cor.correct(&original, &humanized, Tools { rewrite: &mut rewrite, detect: &detect, score_doc: &score_doc, progress: &progress })?;
        let before = det.score(&humanized, doc_type)?["score"].clone();
        let after = det.score(r["text"].as_str().unwrap(), doc_type)?["score"].clone();
        r["scores"] = json!([before, after]);
        Ok(r)
    })();
    *st.task.lock().unwrap() = None;
    c.json(r?, 200);
    Ok(())
}

fn write(c: &mut Conn, req: &Value) -> Result<()> {
    let opening = text::py_strip(s(req, "opening")).to_string();
    let instructions: String = text::py_strip(s(req, "instructions")).chars().take(2000).collect();
    let instructions = if instructions.is_empty() { None } else { Some(instructions) };
    if opening.is_empty() && instructions.is_none() {
        c.json(json!({"error": "Give it an opening sentence or say what it should write."}), 400);
        return Ok(());
    }
    let domain = if WRITE_TYPES.contains(&s(req, "type")) { s(req, "type") } else { "essay" };
    let name = if ["writer_r3", "writer_sft"].contains(&s(req, "model")) { s(req, "model") } else { "writer_r3" };
    let topic = text::py_strip(s(req, "topic"));
    let topic = if domain == "essay" && !topic.is_empty() { Some(topic) } else { None };
    let st = c.state.clone();
    let _g = st.gpu.lock().unwrap();
    st.stop.store(false, Ordering::Relaxed);
    st.ensure_generator(name)?;
    c.start_stream();
    let max = num(req, "max_tokens", 640.0).clamp(32.0, 1024.0) as usize;
    let temperature = num(req, "temperature", 0.9).clamp(0.1, 1.5) as f32;
    stream(c, &generator::write_prompt(domain, &opening, topic, instructions.as_deref()), max, temperature, false)?;
    Ok(())
}

fn update(c: &mut Conn) -> Result<()> {
    let st = c.state.clone();
    st.stop.store(true, Ordering::Relaxed); // finish any writing first, then nothing else runs while files change
    let _g = st.gpu.lock().unwrap();
    c.start_stream();
    let mut say = |line: &str| {
        c.piece(&format!("{line}\n"));
    };
    match updater::apply(&mut say) {
        Ok(()) => {
            c.piece("RESTART\n");
            std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                updater::restart();
            });
        }
        Err(e) => {
            c.piece(&format!("ERROR {e:#}\n"));
        }
    }
    Ok(())
}
