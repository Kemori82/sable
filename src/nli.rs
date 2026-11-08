//! The corrector's meaning checker: cross-encoder/nli-deberta-v3-base (Apache-2.0), as Corrector.nli().
use crate::{onnx, paths};
use anyhow::{anyhow, Result};
use ort::session::Session;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct Nli {
    session: Mutex<Session>,
    tok: tokenizers::Tokenizer,
    pub ent: usize,
    pub con: usize,
    cache: Mutex<HashMap<(String, String), [f32; 3]>>,
}

impl Nli {
    pub fn load() -> Result<Self> {
        let d = paths::models().join("nli");
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json"))?)?;
        let (mut ent, mut con) = (0, 0);
        for (k, v) in cfg["id2label"].as_object().ok_or(anyhow!("no labels"))? {
            match v.as_str().unwrap_or("").to_lowercase().as_str() {
                "entailment" => ent = k.parse()?,
                "contradiction" => con = k.parse()?,
                _ => {}
            }
        }
        Ok(Nli {
            session: Mutex::new(onnx::session(&d.join("model.onnx"))?),
            tok: onnx::tokenizer(&d.join("tokenizer.json"), 512)?,
            ent,
            con,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Probabilities [n][3] for each (premise, hypothesis) pair. The corrector asks about the same pairs many
    /// times while it tries changes, so answers are remembered (the model is deterministic: same answer).
    pub fn probs(&self, premises: &[String], hypotheses: &[String]) -> Result<Vec<[f32; 3]>> {
        onnx::timed("nli", || self.probs_(premises, hypotheses))
    }

    fn probs_(&self, premises: &[String], hypotheses: &[String]) -> Result<Vec<[f32; 3]>> {
        let mut out = vec![[0f32; 3]; premises.len()];
        let mut todo = vec![];
        {
            let c = self.cache.lock().unwrap();
            for (i, (p, h)) in premises.iter().zip(hypotheses).enumerate() {
                match c.get(&(p.clone(), h.clone())) {
                    Some(v) => out[i] = *v,
                    None => todo.push(i),
                }
            }
        }
        for chunk in todo.chunks(16) {
            let pairs: Vec<tokenizers::EncodeInput> = chunk.iter()
                .map(|&i| (premises[i].as_str(), hypotheses[i].as_str()).into()).collect();
            let encs = self.tok.encode_batch(pairs, true).map_err(|e| anyhow!("{e}"))?;
            // token types are ignored by this model (type_vocab_size 0), so the exported graph has no such input
            let (ids, mask, _, len, rows) = onnx::pad_bucketed(&encs, 0);
            let mut s = self.session.lock().unwrap();
            let r = s.run(ort::inputs!["input_ids" => onnx::i64_tensor(rows, len, ids)?,
                                       "attention_mask" => onnx::i64_tensor(rows, len, mask)?])?;
            let (_, lg) = r["logits"].try_extract_tensor::<f32>()?;
            let mut c = self.cache.lock().unwrap();
            for (k, &i) in chunk.iter().enumerate() {
                let l = &lg[k * 3..k * 3 + 3];
                let m = l.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                let p = [e[0] / z, e[1] / z, e[2] / z];
                out[i] = p;
                c.insert((premises[i].clone(), hypotheses[i].clone()), p);
            }
        }
        Ok(out)
    }

    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}
