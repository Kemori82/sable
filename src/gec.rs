//! The corrector's grammar fixer: T5-small (Unbabel/gec-t5_small, or Sable's tuned copy), with the same beam
//! search as Hugging Face's generate(num_beams=4, max_new_tokens=160): length penalty 1, early_stopping False,
//! 2 x beams candidates kept per step, scores in float32 with the same -1e9 masking, so it picks the same fixes.
//!
//! The decoder keeps its attention cache between steps (Hugging Face Optimum's export, weights stored as
//! bfloat16 and run in float32), so each step only reads the newest word.
use crate::{onnx, paths, text};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::sync::Mutex;

const BEAMS: usize = 4;
const KEEP: usize = 2 * BEAMS; // max(2, 1 + number of EOS tokens) * beams
const MAX_NEW: usize = 160;
const EOS: i64 = 1;
const START: i64 = 0;
const NEG: f32 = -1.0e9;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;

pub struct Gec {
    enc: Mutex<Session>,
    dec: Mutex<Session>,
    tok: tokenizers::Tokenizer,
    layers: usize,
    prefix: String,
}

struct Sent {
    running: Vec<(Vec<i64>, f32)>,
    finished: Vec<(Vec<i64>, f32, bool)>,
    unsatisfied: bool,
    done: bool,
}

impl Gec {
    pub fn load(name: &str) -> Result<Self> {
        let d = paths::models().join(name);
        let dec = onnx::cpu_session(&d.join("decoder_model_merged.onnx"))?;
        let layers = dec.inputs().iter().filter(|i| i.name().ends_with(".decoder.key")).count();
        Ok(Gec {
            enc: Mutex::new(onnx::cpu_session(&d.join("encoder_model.onnx"))?),
            dec: Mutex::new(dec),
            tok: onnx::tokenizer(&d.join("tokenizer.json"), 256)?,
            layers,
            prefix: "gec: ".into(),
        })
    }

    /// Each sentence fixed, then given back the original's spacing where only spaces differ (fix_grammar).
    pub fn fix(&self, sents: &[String]) -> Result<Vec<String>> {
        onnx::timed("gec", || self.fix_(sents))
    }

    fn fix_(&self, sents: &[String]) -> Result<Vec<String>> {
        let mut out = vec![];
        for chunk in sents.chunks(16) {
            out.extend(self.generate(chunk)?);
        }
        Ok(sents.iter().zip(out).map(|(s, f)| text::respace(s, &f)).collect())
    }

    fn generate(&self, sents: &[String]) -> Result<Vec<String>> {
        let inputs: Vec<String> = sents.iter().map(|s| format!("{}{s}", self.prefix)).collect();
        let encs = self.tok.encode_batch(inputs, true).map_err(|e| anyhow!("{e}"))?;
        let (ids, mask, _, slen) = onnx::pad(&encs, 0);
        let n = sents.len();
        let (hid, d) = {
            let mut e = self.enc.lock().unwrap();
            let out = e.run(ort::inputs!["input_ids" => onnx::i64_tensor(n, slen, ids)?,
                                         "attention_mask" => onnx::i64_tensor(n, slen, mask.clone())?])?;
            let (shape, h) = out["last_hidden_state"].try_extract_tensor::<f32>()?;
            (h.to_vec(), shape[2] as usize)
        };
        let mut st: Vec<Sent> = (0..n).map(|_| Sent {
            running: (0..BEAMS).map(|b| (vec![START], if b == 0 { 0.0 } else { NEG })).collect(),
            finished: (0..BEAMS).map(|_| (vec![], NEG, false)).collect(),
            unsatisfied: true,
            done: false,
        }).collect();
        let max_length = 1 + MAX_NEW;
        let l = self.layers;
        let enc_kv_size = HEADS * slen * HEAD_DIM; // one row of an encoder key or value
        // per sentence: the cross-attention keys and values (the same for all its beams), filled at the first step
        let mut enc_kv: Vec<Vec<Vec<f32>>> = vec![vec![]; n];
        // per live row (sentence-major, BEAMS each): the self-attention cache, one Vec per layer x {key, value}
        let mut live: Vec<usize> = (0..n).collect();
        let mut past: Vec<Vec<f32>> = vec![vec![]; 2 * l];
        let mut t = 0usize; // cached length
        let mut cur_len = 1usize;
        // per-row encoder inputs, rebuilt only when a sentence finishes (they don't change between steps)
        let mut fixed: Option<(usize, Tensor<i64>, Tensor<f32>, Vec<Tensor<f32>>)> = None;
        // the sentences the batch rows belong to: finished ones stay in (their rows are ignored) until half the
        // batch is done, so the per-row tensors aren't rebuilt every time one sentence finishes
        let mut built: Vec<usize> = live.clone();
        let mut tm = [0f64; 5];
        while !live.is_empty() {
            let t_a = std::time::Instant::now();
            let first = t == 0;
            let rebuild = fixed.as_ref().map(|f| f.0 == usize::MAX).unwrap_or(true) || live.len() * 2 <= built.len();
            if rebuild && !first {
                // drop finished sentences' rows from the cache too
                let row_size = HEADS * t * HEAD_DIM;
                let keep: Vec<usize> = built.iter().enumerate().filter(|(_, i)| live.contains(i)).map(|(k, _)| k).collect();
                for p in past.iter_mut() {
                    let mut v = Vec::with_capacity(keep.len() * BEAMS * row_size);
                    for &k in &keep {
                        v.extend_from_slice(&p[k * BEAMS * row_size..(k + 1) * BEAMS * row_size]);
                    }
                    *p = v;
                }
                built = live.clone();
            }
            let rows = built.len() * BEAMS;
            if rebuild {
                let live = built.clone();
                let mut ehid = Vec::with_capacity(rows * slen * d);
                let mut emask = Vec::with_capacity(rows * slen);
                for &i in &live {
                    for _ in 0..BEAMS {
                        ehid.extend_from_slice(&hid[i * slen * d..(i + 1) * slen * d]);
                        emask.extend_from_slice(&mask[i * slen..(i + 1) * slen]);
                    }
                }
                let mut ekv = vec![];
                for p in 0..2 * l {
                    if first {
                        ekv.push(Tensor::from_array((vec![rows as i64, HEADS as i64, 0, HEAD_DIM as i64], Vec::<f32>::new()))?);
                    } else {
                        let mut v = Vec::with_capacity(rows * enc_kv_size);
                        for &i in &live {
                            for _ in 0..BEAMS {
                                v.extend_from_slice(&enc_kv[i][p]);
                            }
                        }
                        ekv.push(Tensor::from_array((vec![rows as i64, HEADS as i64, slen as i64, HEAD_DIM as i64], v))?);
                    }
                }
                fixed = Some((if first { usize::MAX } else { rows }, onnx::i64_tensor(rows, slen, emask)?,
                              Tensor::from_array((vec![rows as i64, slen as i64, d as i64], ehid))?, ekv));
            }
            tm[0] += t_a.elapsed().as_secs_f64();
            let t_b = std::time::Instant::now();
            let dec_ids: Vec<i64> = built.iter().flat_map(|&i| (0..BEAMS).map(move |b| (i, b)))
                .map(|(i, b)| *st[i].running[b].0.last().unwrap()).collect();
            let f = fixed.as_ref().unwrap();
            let mut inputs: Vec<(String, ort::session::SessionInputValue)> = vec![
                ("encoder_attention_mask".into(), (&f.1).into()),
                ("input_ids".into(), onnx::i64_tensor(rows, 1, dec_ids)?.into()),
                ("encoder_hidden_states".into(), (&f.2).into()),
                ("use_cache_branch".into(), Tensor::from_array((vec![1i64], vec![!first]))?.into()),
            ];
            for layer in 0..l {
                for (k, kind) in ["key", "value"].iter().enumerate() {
                    let dshape = vec![rows as i64, HEADS as i64, t as i64, HEAD_DIM as i64];
                    let data = if first { vec![] } else { std::mem::take(&mut past[2 * layer + k]) };
                    inputs.push((format!("past_key_values.{layer}.decoder.{kind}"), Tensor::from_array((dshape, data))?.into()));
                    inputs.push((format!("past_key_values.{layer}.encoder.{kind}"), (&f.3[2 * layer + k]).into()));
                }
            }
            tm[1] += t_b.elapsed().as_secs_f64();
            let t_c = std::time::Instant::now();
            // run the step, then work straight from its outputs (no copies of the logits or the cache)
            let mut s = self.dec.lock().unwrap();
            let out = onnx::timed("gec decoder run", || s.run(inputs))?;
            let (shape, logits) = out["logits"].try_extract_tensor::<f32>()?;
            let vocab = shape[2] as usize;
            if first {
                let mut encp = vec![];
                for layer in 0..l {
                    for kind in ["key", "value"] {
                        encp.push(out[format!("present.{layer}.encoder.{kind}").as_str()].try_extract_tensor::<f32>()?.1);
                    }
                }
                for (k, &i) in built.iter().enumerate() {
                    let row = k * BEAMS;
                    enc_kv[i] = encp.iter().map(|p| p[row * enc_kv_size..(row + 1) * enc_kv_size].to_vec()).collect();
                }
            }
            tm[2] += t_c.elapsed().as_secs_f64();
            let t_d = std::time::Instant::now();
            // each live sentence's beam step, side by side
            let mut sources: Vec<Vec<usize>> = vec![vec![]; n];
            {
                let pos: std::collections::HashMap<usize, usize> = built.iter().enumerate().map(|(k, &i)| (i, k)).collect();
                let mut work: Vec<(usize, &mut Sent)> = st.iter_mut().enumerate().filter(|(i, _)| live.contains(i)).collect();
                let res: Vec<(usize, Vec<usize>)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = work.iter_mut().map(|(i, s)| {
                        let k = pos[i];
                        let i = *i;
                        sc.spawn(move || (i, step(s, &logits[k * BEAMS * vocab..(k + 1) * BEAMS * vocab], vocab, cur_len, max_length)))
                    }).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for (i, src) in res {
                    sources[i] = src;
                }
            }
            cur_len += 1;
            for &i in &live {
                let s = &mut st[i];
                if s.done {
                    continue;
                }
                // early-stop heuristic (early_stopping=False): best running score at the current length
                let best = s.running[0].1 / (cur_len - 1) as f32;
                let min_score = s.finished.iter().map(|f| f.1).fold(f32::INFINITY, f32::min);
                s.unsatisfied &= s.finished.iter().any(|f| best > if f.2 { min_score } else { NEG });
                if !s.unsatisfied {
                    s.done = true;
                }
            }
            tm[3] += t_d.elapsed().as_secs_f64();
            let t_e = std::time::Instant::now();
            // next step's cache: the rows of the beams each sentence continues from
            let row_size = HEADS * (t + 1) * HEAD_DIM;
            let new_live: Vec<usize> = live.iter().copied().filter(|&i| !st[i].done).collect();
            let mut p = 0;
            for layer in 0..l {
                for kind in ["key", "value"] {
                    let pres = out[format!("present.{layer}.decoder.{kind}").as_str()].try_extract_tensor::<f32>()?.1;
                    let mut v = Vec::with_capacity(built.len() * BEAMS * row_size);
                    for (k, &i) in built.iter().enumerate() {
                        for b in 0..BEAMS {
                            // a live sentence's beams continue from their source beams; a finished one's are left as they are
                            let src = if new_live.contains(&i) { sources[i][b] } else { b };
                            let row = k * BEAMS + src;
                            v.extend_from_slice(&pres[row * row_size..(row + 1) * row_size]);
                        }
                    }
                    past[p] = v;
                    p += 1;
                }
            }
            drop(out);
            drop(s);
            tm[4] += t_e.elapsed().as_secs_f64();
            live = new_live;
            t += 1;
        }
        if std::env::var("SABLE_DEBUG").is_ok() { eprintln!("gec parts: rebuild {:.2} inputs {:.2} run {:.2} beams {:.2} gather {:.2}", tm[0], tm[1], tm[2], tm[3], tm[4]); }
        Ok(st.iter().map(|s| {
            let ids: Vec<u32> = s.finished[0].0.iter().filter(|t| **t > 2).map(|t| *t as u32).collect();
            self.tok.decode(&ids, true).unwrap_or_default().trim().to_string()
        }).collect())
    }
}

/// One beam-search step for one sentence (HF _get_top_k_continuations, _get_running_beams_for_next_iteration,
/// _update_finished_beams). Returns, for each new running beam, the beam it continues.
fn step(s: &mut Sent, logits: &[f32], vocab: usize, cur_len: usize, max_length: usize) -> Vec<usize> {
    // log_softmax per beam ((x - max) - log(sum(exp(x - max))), as torch does), plus the beam's running score.
    // Within a beam the order is the raw logits' order, so the best KEEP overall are among each beam's best KEEP.
    let mut pool: Vec<(f32, usize)> = Vec::with_capacity(BEAMS * KEEP);
    for b in 0..BEAMS {
        let l = &logits[b * vocab..(b + 1) * vocab];
        let m = l.iter().cloned().fold(f32::MIN, f32::max);
        let lse = l.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.select_nth_unstable_by(KEEP - 1, |x, y| l[*y].partial_cmp(&l[*x]).unwrap().then(x.cmp(y)));
        for &v in &idx[..KEEP] {
            pool.push((((l[v] - m) - lse) + s.running[b].1, b * vocab + v));
        }
    }
    pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let cands: Vec<(Vec<i64>, f32, bool, usize)> = pool[..KEEP].iter().map(|&(score, k)| {
        let (beam, tok) = (k / vocab, (k % vocab) as i64);
        let mut seq = s.running[beam].0.clone();
        seq.push(tok);
        let hits = tok == EOS || cur_len + 1 >= max_length;
        (seq, score, hits, beam)
    }).collect();
    if cands.iter().all(|c| c.2) {
        s.done = true; // no valid continuation (e.g. max length)
    }
    // next running beams: the best BEAMS that didn't just finish
    let mut run: Vec<(usize, f32)> = cands.iter().enumerate().map(|(i, c)| (i, c.1 + if c.2 { NEG } else { 0.0 })).collect();
    run.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let sources: Vec<usize> = run[..BEAMS].iter().map(|(i, _)| cands[*i].3).collect();
    let new_running: Vec<(Vec<i64>, f32)> = run[..BEAMS].iter().map(|(i, sc)| (cands[*i].0.clone(), *sc)).collect();
    // finished: only the first BEAMS candidates can become final; length penalty 1 over the generated length
    let mut merged: Vec<(Vec<i64>, f32, bool)> = s.finished.clone();
    for (i, c) in cands.iter().enumerate() {
        let just = c.2 && i < BEAMS;
        let mut sc = c.1 / cur_len as f32;
        if !s.unsatisfied {
            sc += NEG;
        }
        if !just {
            sc += NEG;
        }
        merged.push((c.0.clone(), sc, just));
    }
    let mut order: Vec<usize> = (0..merged.len()).collect();
    order.sort_by(|a, b| merged[*b].1.partial_cmp(&merged[*a].1).unwrap().then(a.cmp(b)));
    s.finished = order[..BEAMS].iter().map(|i| merged[*i].clone()).collect();
    s.running = new_running;
    sources
}
