//! Development checks against the Python version: run one part of Sable over a JSONL file and write its outputs
//! for parity/compare.py.
//!
//!   Sable.exe parity text      in.jsonl out.jsonl   normalize, segments, passages, sentences, repeats, diffs
//!   Sable.exe parity detector  in.jsonl out.jsonl   score(), segment logits, passage_prob
//!   Sable.exe parity nli       in.jsonl out.jsonl   probabilities per (premise, hypothesis)
//!   Sable.exe parity gec NAME  in.jsonl out.jsonl   grammar fixes (gec or gec-tuned)
//!   Sable.exe parity humanize MODEL TRIES in.jsonl out.jsonl   rewrites (tries > 1: best by the detector)
//!   Sable.exe parity bench                          timings
use crate::{detector::Detector, gec::Gec, generator, llm::Llm, nli::Nli, text};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// models.json for every file in models/ (SHA-256, size, release asset name: path with "/" -> "--"), and the
/// release assets themselves in OUT_DIR, files over 1.9 GB split into numbered parts (GitHub allows 2 GB each).
///   Sable.exe parity manifest OWNER/REPO TAG OUT_DIR
fn manifest(repo: &str, tag: &str, out_dir: &str) -> Result<()> {
    const LIMIT: u64 = 1_900_000_000;
    let models = crate::paths::models();
    let out = std::path::Path::new(out_dir);
    std::fs::create_dir_all(out)?;
    let mut files = vec![];
    let mut stack = vec![models.clone()];
    let mut all = vec![];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)? {
            let p = e?.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                all.push(p);
            }
        }
    }
    all.sort();
    for p in all {
        let rel = p.strip_prefix(&models)?.to_string_lossy().replace('\\', "/");
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with('.') || name.ends_with(".bak") || name.ends_with(".hidden") || name.contains(".part") {
            continue;
        }
        let size = std::fs::metadata(&p)?.len();
        let asset = rel.replace('/', "--");
        let mut e = json!({"path": rel, "size": size, "sha256": crate::install::sha256(&p)?});
        if size > LIMIT {
            let mut f = std::fs::File::open(&p)?;
            let (mut names, mut sizes) = (vec![], vec![]);
            let mut i = 0;
            loop {
                let n = format!("{asset}.part{i}");
                let mut w = std::fs::File::create(out.join(&n))?;
                let got = std::io::copy(&mut std::io::Read::take(&mut f, LIMIT), &mut w)?;
                if got == 0 {
                    drop(w);
                    std::fs::remove_file(out.join(&n))?;
                    break;
                }
                names.push(n);
                sizes.push(got);
                i += 1;
            }
            e["assets"] = json!(names);
            e["asset_sizes"] = json!(sizes);
        } else {
            std::fs::copy(&p, out.join(&asset))?;
            e["asset"] = json!(asset);
        }
        eprintln!("{rel} {size}");
        files.push(e);
    }
    let m = json!({"repo": repo, "tag": tag, "files": files});
    std::fs::write(crate::paths::app().join("models.json.new"), serde_json::to_string_pretty(&m)?)?;
    eprintln!("wrote models.json.new; upload everything in {out_dir} to the {tag} release");
    Ok(())
}

fn rows(path: &str) -> Result<Vec<Value>> {
    Ok(std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?)
}

fn write(path: &str, out: &[Value]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in out {
        writeln!(f, "{v}")?;
    }
    Ok(())
}

fn st(v: &Value, k: &str) -> String {
    v[k].as_str().unwrap_or("").to_string()
}

pub fn main(args: &[String]) -> Result<()> {
    let a = |i: usize| args.get(i).cloned().ok_or(anyhow!("missing argument {i}"));
    match a(0)?.as_str() {
        "text" => {
            let mut out = vec![];
            for r in rows(&a(1)?)? {
                let t = st(&r, "text");
                let n = text::normalize(&t);
                let segs: Vec<Value> = text::segment(&n).iter().map(|sp| json!(text::char_span(&n, *sp))).collect();
                let hum = st(&r, "rewrite");
                out.push(json!({"normalize": n, "segments": segs, "chunks": text::chunks(&t, 4),
                                "sentences": text::sentences(&t), "paragraphs": text::paragraphs(&t),
                                "unrepeat": text::unrepeat(&t), "diff": text::diff(&t, &hum),
                                "respace": text::respace(&t, &hum), "safe": text::safe(&t, &hum),
                                "small_edit": text::small_edit(&t, &hum)}));
            }
            write(&a(2)?, &out)
        }
        "detector" => {
            let d = Detector::load()?;
            let mut out = vec![];
            let t0 = Instant::now();
            for r in rows(&a(1)?)? {
                let ty = st(&r, "type");
                let doc = if ty == "essay" || ty == "abstract" { ty.as_str() } else { "other" };
                let mut s = d.score(&st(&r, "text"), doc)?;
                s["seg_logits"] = json!(d.segment_logits(&st(&r, "text"))?);
                s["passage_prob"] = json!(d.passage_prob(&[st(&r, "text")])?[0]);
                s["id"] = r["id"].clone();
                out.push(s);
            }
            eprintln!("detector: {} texts in {:.1}s", out.len(), t0.elapsed().as_secs_f64());
            write(&a(2)?, &out)
        }
        "nli" => {
            let n = Nli::load()?;
            let rs = rows(&a(1)?)?;
            let t0 = Instant::now();
            let p = n.probs(&rs.iter().map(|r| st(r, "premise")).collect::<Vec<_>>(), &rs.iter().map(|r| st(r, "hypothesis")).collect::<Vec<_>>())?;
            eprintln!("nli: {} pairs in {:.1}s", rs.len(), t0.elapsed().as_secs_f64());
            write(&a(2)?, &p.iter().map(|x| json!(x)).collect::<Vec<_>>())
        }
        "gec" => {
            let g = Gec::load(&a(1)?)?;
            let rs = rows(&a(2)?)?;
            let w = Instant::now();
            g.fix(&rs.iter().take(16).map(|r| st(r, "src")).collect::<Vec<_>>())?; // warm-up, as in the Python timing
            eprintln!("warm-up {:.1}s", w.elapsed().as_secs_f64());
            crate::onnx::TIMES.lock().unwrap().clear();
            let t0 = Instant::now();
            let f = g.fix(&rs.iter().map(|r| st(r, "src")).collect::<Vec<_>>())?;
            eprintln!("gec: {} sentences in {:.1}s", rs.len(), t0.elapsed().as_secs_f64());
            crate::onnx::report();
            write(&a(3)?, &f.iter().map(|x| json!(x)).collect::<Vec<_>>())
        }
        "humanize" => {
            let (model, tries) = (a(1)?, a(2)?.parse::<usize>()?);
            let mut llm = Llm::load(&|m| eprintln!("{m}"))?;
            llm.use_chain(&generator::chain(&model).ok_or(anyhow!("model"))?, &|_| {})?;
            let det = if tries > 1 { Some(Detector::load()?) } else { None };
            let stop = AtomicBool::new(false);
            let mut out = vec![];
            let t0 = Instant::now();
            let mut tokens = 0usize;
            for r in rows(&a(3)?)? {
                let what = generator::what(&st(&r, "domain"));
                let mut parts = vec![];
                for (i, para) in text::chunks(&st(&r, "text"), 4).iter().enumerate() {
                    for (j, p) in para.iter().enumerate() {
                        if i > 0 || j > 0 {
                            parts.push(if j == 0 { "\n\n".to_string() } else { " ".to_string() });
                        }
                        let max = (llm.n_tokens(p) as f64 * 1.8) as usize + 32;
                        let chat = Llm::chat(&generator::humanize_prompt(what, p));
                        let outs = llm.generate(&vec![chat; tries], &vec![max; tries], 0.8, &stop, None)?;
                        tokens += outs.iter().map(|o| llm.n_tokens(o)).sum::<usize>();
                        let cands: Vec<String> = outs.iter().map(|o| text::py_strip(o).to_string()).filter(|o| !o.is_empty()).collect();
                        let words = text::n_words(p).max(1) as f64;
                        let ok: Vec<String> = cands.iter().filter(|c| (0.7..=1.5).contains(&(text::n_words(c) as f64 / words))).cloned().collect();
                        let ok = if ok.is_empty() { cands } else { ok };
                        let best = match &det {
                            Some(d) if !ok.is_empty() => {
                                let pr = d.passage_prob(&ok)?;
                                let mut z: Vec<(f64, &String)> = pr.into_iter().zip(&ok).collect();
                                z.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                                z[0].1.clone()
                            }
                            _ => ok.first().cloned().unwrap_or_default(),
                        };
                        parts.push(best);
                    }
                }
                out.push(json!({"doc_id": r["doc_id"], "domain": r["domain"], "text": r["text"], "output": parts.concat()}));
                eprintln!("{} done ({:.0}s)", out.len(), t0.elapsed().as_secs_f64());
            }
            eprintln!("humanize: {} texts in {:.1}s, {} tokens ({:.0}/s), base {}", out.len(), t0.elapsed().as_secs_f64(),
                      tokens, tokens as f64 / t0.elapsed().as_secs_f64(), llm.base_file);
            write(&a(4)?, &out)
        }
        "manifest" => manifest(&a(1)?, &a(2)?, &a(3)?),
        "generate" => {
            // {id, prompt, n?, temperature?, max_new?} rows -> {id, outputs: [...]}; CHAIN is a comma-separated list
            // of adapters, or "none" (the base model as it is; SABLE_LLM_FILE picks another GGUF)
            let chain_arg = a(1)?;
            let mut llm = Llm::load(&|m| eprintln!("{m}"))?;
            let chain: Vec<&str> = if chain_arg == "none" { vec![] } else { chain_arg.split(',').collect() };
            llm.use_chain(&chain, &|_| {})?;
            let stop = AtomicBool::new(false);
            let rs = rows(&a(2)?)?;
            let t0 = Instant::now();
            let mut out = vec![];
            for (k, r) in rs.iter().enumerate() {
                let n = r["n"].as_u64().unwrap_or(1).clamp(1, 8) as usize;
                let temp = r["temperature"].as_f64().unwrap_or(0.7) as f32;
                let max = r["max_new"].as_u64().unwrap_or(200) as usize;
                let chat = Llm::chat(&st(r, "prompt"));
                let o = llm.generate(&vec![chat; n], &vec![max; n], temp, &stop, None)?;
                out.push(json!({"id": r["id"], "outputs": o}));
                if k % 50 == 49 {
                    eprintln!("{} of {} ({:.0}s)", k + 1, rs.len(), t0.elapsed().as_secs_f64());
                }
            }
            eprintln!("generate: {} prompts in {:.1}s", out.len(), t0.elapsed().as_secs_f64());
            write(&a(3)?, &out)
        }
        "correct" => {
            // {original, humanized, type} rows, corrected with Writer voice repairs; TUNED = gec-tuned + safety check;
            // a 4th argument "repair" also fixes sentences in place with the repair model
            let tuned = a(1)? == "tuned";
            let use_repair = args.get(4).map(|s| s == "repair").unwrap_or(false);
            let det = Detector::load()?;
            let cor = crate::corrector::Corrector { nli: Nli::load()?, gec: Gec::load(if tuned { "gec-tuned" } else { "gec" })?, safe_check: tuned };
            let mut llm = Llm::load(&|_| {})?;
            llm.use_chain(&generator::chain("humanizer_writer").unwrap(), &|_| {})?;
            let llm = std::cell::RefCell::new(llm);
            let stop = AtomicBool::new(false);
            let mut out = vec![];
            let t0 = Instant::now();
            for r in rows(&a(2)?)? {
                let kind = st(&r, "type");
                let what = generator::what(&kind);
                let doc = if kind == "essay" || kind == "abstract" { kind.clone() } else { "other".into() };
                let mut rewrite = |p: &str| -> Result<Vec<String>> { crate::onnx::timed("rewrite", || {
                    let mut l = llm.borrow_mut();
                    l.use_chain(&generator::chain("humanizer_writer").unwrap(), &|_| {})?;
                    let max = (l.n_tokens(p) as f64 * 1.8) as usize + 32;
                    let outs = l.generate(&vec![Llm::chat(&generator::humanize_prompt(what, p)); 8], &[max; 8], 0.8, &stop, None)?;
                    let cands: Vec<String> = outs.iter().map(|o| text::py_strip(o).to_string()).filter(|o| !o.is_empty()).collect();
                    let words = text::n_words(p).max(1) as f64;
                    let ok: Vec<String> = cands.iter().filter(|c| (0.7..=1.5).contains(&(text::n_words(c) as f64 / words))).cloned().collect();
                    Ok(if ok.is_empty() { cands } else { ok })
                }) };
                let mut fix_fn = |op: &str, os: &str, marked: &str| -> Result<Vec<String>> { crate::onnx::timed("repair", || {
                    let mut l = llm.borrow_mut();
                    l.use_chain(&generator::chain("repair").unwrap(), &|_| {})?;
                    let max = (l.n_tokens(os) as f64 * 1.8) as usize + 32;
                    l.generate(&vec![Llm::chat(&generator::repair_prompt(op, os, marked)); 8], &[max; 8], 0.8, &stop, None)
                }) };
                let fix: Option<&mut dyn FnMut(&str, &str, &str) -> Result<Vec<String>>> = if use_repair { Some(&mut fix_fn) } else { None };
                let detect = |t: &[String]| crate::onnx::timed("detect", || det.passage_prob(t));
                let score_doc = |t: &str| -> Result<f64> { crate::onnx::timed("score_doc", || Ok(det.score(t, &doc)?["score"].as_f64().unwrap_or(0.0))) };
                let t1 = Instant::now();
                let mut res = cor.correct(&st(&r, "original"), &st(&r, "humanized"),
                    crate::corrector::Tools { rewrite: &mut rewrite, fix, detect: &detect, score_doc: &score_doc, progress: &|_| {} })?;
                res["seconds"] = json!(t1.elapsed().as_secs_f64());
                res["id"] = r["id"].clone();
                out.push(res);
            }
            eprintln!("correct: {} texts in {:.1}s", out.len(), t0.elapsed().as_secs_f64());
            crate::onnx::report();
            write(&a(3)?, &out)
        }
        other => Err(anyhow!("unknown parity check {other}")),
    }
}
