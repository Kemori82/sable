//! AI-likeness of a text: the DeBERTa + RAID + papers detector with its calibration, as in detector.py.
use crate::{onnx, paths, text};
use anyhow::{anyhow, Result};
use ort::session::Session;
use serde_json::{json, Value};
use std::sync::Mutex;

pub const UNSURE_PERCENTILE: f64 = 90.0; // orange from here: scores higher than 90% of human papers

pub struct Detector {
    session: Mutex<Session>,
    tok: tokenizers::Tokenizer,
    cal: Value,
    ood_ref: Vec<f32>, // [n, 768], rows of length 1
    ood_k: usize,
    ood_cutoff: f64,
}

struct Out {
    logits: Vec<f32>,
    cls: Vec<Vec<f32>>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// numpy.interp for increasing xp
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let i = xp.partition_point(|v| *v <= x); // xp[i-1] <= x < xp[i]
    let (x0, x1, y0, y1) = (xp[i - 1], xp[i], fp[i - 1], fp[i]);
    if x1 == x0 { y0 } else { y0 + (x - x0) * (y1 - y0) / (x1 - x0) }
}

/// .npy reader for the OOD reference (float16 or float32, C order).
fn read_npy(path: &std::path::Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let b = std::fs::read(path)?;
    let hlen = if b[6] == 1 { u16::from_le_bytes([b[8], b[9]]) as usize + 10 } else { u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize + 12 };
    let header = String::from_utf8_lossy(&b[8..hlen]); // after the "\x93NUMPY" magic and version
    let shape: Vec<usize> = header.split("'shape': (").nth(1).ok_or(anyhow!("npy shape"))?.split(')').next().unwrap()
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let data = &b[hlen..];
    let v: Vec<f32> = if header.contains("<f2") {
        data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
    } else if header.contains("<f4") {
        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else {
        return Err(anyhow!("unsupported npy dtype"));
    };
    if header.contains("'fortran_order': True") && shape.len() == 2 {
        // stored column by column: put it in row order
        let (r, c) = (shape[0], shape[1]);
        let mut t = vec![0f32; r * c];
        for j in 0..c {
            for i in 0..r {
                t[i * c + j] = v[j * r + i];
            }
        }
        return Ok((t, shape));
    }
    Ok((v, shape))
}

impl Detector {
    pub fn load() -> Result<Self> {
        let d = paths::models().join("detector");
        let ood: Value = serde_json::from_str(&std::fs::read_to_string(d.join("ood.json"))?)?;
        let (ood_ref, _) = read_npy(&d.join("ood_reference.npy"))?;
        Ok(Detector {
            session: Mutex::new(onnx::session(&d.join("model.onnx"))?),
            tok: onnx::tokenizer(&d.join("tokenizer.json"), 256)?,
            cal: serde_json::from_str(&std::fs::read_to_string(d.join("calibration.json"))?)?,
            ood_ref,
            ood_k: ood["k"].as_u64().unwrap_or(10) as usize,
            ood_cutoff: ood["cutoff"].as_f64().unwrap_or(1.0),
        })
    }

    fn deberta(&self, texts: &[String]) -> Result<Out> {
        let mut logits = Vec::with_capacity(texts.len());
        let mut cls = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(32) {
            let encs = self.tok.encode_batch(chunk.to_vec(), true).map_err(|e| anyhow!("{e}"))?;
            let (ids, mask, _, len, rows) = onnx::pad_bucketed(&encs, 0);
            let mut s = self.session.lock().unwrap();
            let (ids, mask) = (onnx::i64_tensor(rows, len, ids)?, onnx::i64_tensor(rows, len, mask)?);
            let out = onnx::timed("detector", || s.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask]))?;
            let (_, lg) = out["logit"].try_extract_tensor::<f32>()?;
            logits.extend_from_slice(&lg[..chunk.len()]);
            let (shape, c) = out["cls"].try_extract_tensor::<f32>()?;
            let h = shape[1] as usize;
            cls.extend(c.chunks(h).take(chunk.len()).map(|r| r.to_vec()));
        }
        Ok(Out { logits, cls })
    }

    fn standardization(&self, which: &str) -> (f64, f64) {
        let name = self.cal["models"]["deberta"].as_str().unwrap_or("");
        let v = &self.cal[which][name];
        (v["mean"].as_f64().unwrap(), v["std"].as_f64().unwrap())
    }

    fn platt(&self, s: f64) -> f64 {
        let p = &self.cal["platt"];
        sigmoid(p["a"].as_f64().unwrap() * s + p["b"].as_f64().unwrap())
    }

    /// AI-likeness of short passages (one segment each, no minimum length): for choosing between rewrites.
    pub fn passage_prob(&self, texts: &[String]) -> Result<Vec<f64>> {
        let norm: Vec<String> = texts.iter().map(|t| text::normalize(t)).collect();
        let out = self.deberta(&norm)?;
        let (m, sd) = self.standardization("doc_standardization");
        Ok(out.logits.iter().map(|lg| self.platt((*lg as f64 - m) / sd)).collect())
    }

    /// Score, band (green / orange / red), highlighted segments and notes, as in Detector.score().
    pub fn score(&self, raw: &str, doc_type: &str) -> Result<Value> {
        let cal = &self.cal;
        let t = text::normalize(raw);
        let words = text::py_split(&t);
        let mut res = json!({"doc_type": doc_type, "word_count": words.len(), "text": t, "notes": []});
        let min_words = cal["min_words"].as_u64().unwrap_or(50) as usize;
        if words.len() < min_words {
            res["status"] = json!("too_short");
            res["notes"] = json!([format!("Fewer than {min_words} words: too short to score.")]);
            return Ok(res);
        }
        if text::english_share(&words) < 0.08 {
            res["status"] = json!("unsupported_language");
            res["notes"] = json!(["Only English text can be scored."]);
            return Ok(res);
        }
        let spans = text::segment(&t);
        let seg_texts: Vec<String> = spans.iter().map(|(a, b)| t[*a..*b].to_string()).collect();
        let out = self.deberta(&seg_texts)?;
        let w: Vec<f64> = seg_texts.iter().map(|s| text::n_words(s) as f64).collect();
        let wsum: f64 = w.iter().sum();
        let mean_logit = out.logits.iter().zip(&w).map(|(l, w)| *l as f64 * w).sum::<f64>() / wsum;
        let (dm, ds) = self.standardization("doc_standardization");
        let (sm, ss) = self.standardization("segment_standardization");
        let s = (mean_logit - dm) / ds;
        let prob = self.platt(s);
        let domain = if doc_type == "essay" || doc_type == "abstract" { doc_type } else { "essay" };
        let r = &cal["human_percentile_reference"];
        let xp: Vec<f64> = r["scores"][domain].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let fp: Vec<f64> = r["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let pct = interp(s, &xp, &fp);
        let bands = &cal["bands"];
        let (unsure, review) = (bands["unsure_prob"].as_f64().unwrap(), bands["review_prob"].as_f64().unwrap());
        let band = if prob >= review { "red" } else if prob >= unsure || pct >= UNSURE_PERCENTILE { "orange" } else { "green" };

        let mut notes = vec![];
        // out-of-distribution: mean [CLS] vector far from the training documents'
        let dim = out.cls[0].len();
        let mut v = vec![0f32; dim];
        for c in &out.cls {
            for (a, b) in v.iter_mut().zip(c) {
                *a += b;
            }
        }
        let n = out.cls.len() as f32;
        v.iter_mut().for_each(|a| *a /= n);
        let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        let mut sims: Vec<f32> = self.ood_ref.chunks(dim).map(|r| r.iter().zip(&v).map(|(a, b)| a * b / norm).sum()).collect();
        sims.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top = sims[..self.ood_k.min(sims.len())].iter().sum::<f32>() / self.ood_k as f32;
        if std::env::var("SABLE_DEBUG").is_ok() {
            eprintln!("ood: v[..5] {:?} norm {norm} ref[0][..5] {:?} dist {}", &v[..5], &self.ood_ref[..5], 1.0 - top);
        }
        if (1.0 - top) as f64 > self.ood_cutoff {
            notes.push("This text is unlike the writing the detector was trained on, so the score is less reliable.".to_string());
        }
        let short = cal["short_text_words"].as_u64().unwrap_or(150) as usize;
        if words.len() < short {
            notes.push(format!("Short text (under {short} words): scores are noisier."));
        }
        if text::has_confusables(raw) {
            notes.push("Look-alike characters (e.g. Cyrillic letters) were converted to Latin before scoring.".to_string());
        }
        let cutoff = cal["segment_highlight_cutoff"].as_f64().unwrap();
        let segments: Vec<Value> = spans.iter().zip(&out.logits).map(|(sp, lg)| {
            let (a, b) = text::char_span(&t, *sp);
            json!({"start": a, "end": b, "highlight": (*lg as f64 - sm) / ss >= cutoff})
        }).collect();
        res["status"] = json!("scored");
        res["score"] = json!(prob);
        res["band"] = json!(band);
        res["human_percentile"] = json!(pct);
        res["unsure_from"] = json!(unsure);
        res["review_from"] = json!(review);
        res["notes"] = json!(notes);
        res["segments"] = json!(segments);
        Ok(res)
    }

    /// Raw segment logits (parity checks).
    pub fn segment_logits(&self, raw: &str) -> Result<Vec<f32>> {
        let t = text::normalize(raw);
        let segs: Vec<String> = text::segment(&t).iter().map(|(a, b)| t[*a..*b].to_string()).collect();
        if segs.is_empty() {
            return Ok(vec![]);
        }
        Ok(self.deberta(&segs)?.logits)
    }
}
