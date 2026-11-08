//! The writer and humanizer models on llama.cpp: one copy of the base Qwen2.5-1.5B-Instruct on the graphics card
//! (Vulkan: NVIDIA, AMD, Intel) or the CPU, with each model's chain of LoRA adapters applied on top. Every Sable
//! model is the base plus a chain of adapters, so switching models only swaps adapters: nothing reloads.
//!
//! Sampling is as in training: softmax(logits / temperature), no top-k, top-p or repetition penalty.
use crate::paths;
use anyhow::{anyhow, Result};
use llama_cpp_sys_2 as ll;
use rand::Rng;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};

const N_SEQ: u32 = 8;
const N_CTX: u32 = 16384;
const N_BATCH: u32 = 2048;
pub const BASE_FILES: [&str; 2] = ["base-q8_0.gguf", "base-bf16.gguf"];

pub struct Llm {
    model: *mut ll::llama_model,
    ctx: *mut ll::llama_context,
    vocab: *const ll::llama_vocab,
    adapters: HashMap<String, *mut ll::llama_adapter_lora>,
    active: Vec<String>,
    tok: tokenizers::Tokenizer,
    n_vocab: usize,
    pub device: String,
    pub base_file: String,
}

unsafe impl Send for Llm {}

unsafe extern "C" fn quiet(_level: ll::ggml_log_level, _text: *const c_char, _data: *mut c_void) {}

/// The graphics card llama.cpp writes on (None: the CPU).
pub fn gpu_name() -> Option<String> {
    if std::env::var("SABLE_CPU").is_ok() {
        return None;
    }
    unsafe {
        for i in 0..ll::ggml_backend_dev_count() {
            let d = ll::ggml_backend_dev_get(i);
            if ll::ggml_backend_dev_type(d) == ll::GGML_BACKEND_DEVICE_TYPE_GPU {
                return Some(CStr::from_ptr(ll::ggml_backend_dev_description(d)).to_string_lossy().into_owned());
            }
        }
    }
    None
}

fn device_name() -> String {
    gpu_name().unwrap_or_else(|| "CPU".into())
}

impl Llm {
    pub fn load(progress: &dyn Fn(&str)) -> Result<Self> {
        let dir = paths::models().join("qwen");
        // SABLE_LLM_FILE: another GGUF model (development: e.g. a bigger teacher model for training data)
        let base = match std::env::var("SABLE_LLM_FILE") {
            Ok(f) => std::path::PathBuf::from(f),
            Err(_) => BASE_FILES.iter().map(|f| dir.join(f)).find(|p| p.exists()).ok_or(anyhow!("base model missing"))?,
        };
        unsafe {
            if std::env::var("SABLE_LLAMA_LOG").is_err() {
                ll::llama_log_set(Some(quiet), std::ptr::null_mut());
            }
            ll::llama_backend_init();
            progress("loading the base model");
            let mut mp = ll::llama_model_default_params();
            mp.n_gpu_layers = if std::env::var("SABLE_CPU").is_ok() { 0 } else { 999 };
            let path = CString::new(base.to_string_lossy().as_bytes())?;
            let model = ll::llama_model_load_from_file(path.as_ptr(), mp);
            if model.is_null() {
                return Err(anyhow!("could not load {}", base.display()));
            }
            let mut cp = ll::llama_context_default_params();
            cp.n_ctx = N_CTX;
            cp.n_batch = N_BATCH;
            cp.n_ubatch = 512;
            cp.n_seq_max = N_SEQ;
            cp.kv_unified = true;
            cp.no_perf = true;
            let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
            cp.n_threads = threads;
            cp.n_threads_batch = threads;
            let ctx = ll::llama_init_from_model(model, cp);
            if ctx.is_null() {
                ll::llama_model_free(model);
                return Err(anyhow!("could not create the model context"));
            }
            let vocab = ll::llama_model_get_vocab(model);
            let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
            Ok(Llm {
                model,
                ctx,
                vocab,
                adapters: HashMap::new(),
                active: vec![],
                tok,
                n_vocab: ll::llama_vocab_n_tokens(vocab) as usize,
                device: device_name(),
                base_file: base.file_name().unwrap().to_string_lossy().into_owned(),
            })
        }
    }

    /// Apply a model's chain of adapters (all at once: the trained deltas add up).
    pub fn use_chain(&mut self, chain: &[&str], progress: &dyn Fn(&str)) -> Result<()> {
        let want: Vec<String> = chain.iter().map(|s| s.to_string()).collect();
        if want == self.active {
            return Ok(());
        }
        for (i, a) in chain.iter().enumerate() {
            if !self.adapters.contains_key(*a) {
                progress(&format!("adapter {}/{}", i + 1, chain.len()));
                let p = paths::models().join("qwen").join("adapters").join(format!("{a}.gguf"));
                let cp = CString::new(p.to_string_lossy().as_bytes())?;
                let ad = unsafe { ll::llama_adapter_lora_init(self.model, cp.as_ptr()) };
                if ad.is_null() {
                    return Err(anyhow!("could not load adapter {a}"));
                }
                self.adapters.insert(a.to_string(), ad);
            }
        }
        let mut ptrs: Vec<*mut ll::llama_adapter_lora> = chain.iter().map(|a| self.adapters[*a]).collect();
        let mut scales = vec![1.0f32; ptrs.len()]; // each adapter's alpha / r is stored in its file
        let rc = unsafe { ll::llama_set_adapters_lora(self.ctx, ptrs.as_mut_ptr(), ptrs.len(), scales.as_mut_ptr()) };
        if rc != 0 {
            return Err(anyhow!("could not apply adapters ({rc})"));
        }
        self.active = want;
        Ok(())
    }

    pub fn tokens(&self, text: &str) -> Vec<i32> {
        self.tok.encode(text, true).map(|e| e.get_ids().iter().map(|t| *t as i32).collect()).unwrap_or_default()
    }

    pub fn n_tokens(&self, text: &str) -> usize {
        self.tok.encode(text, false).map(|e| e.get_ids().len()).unwrap_or(0)
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let ids: Vec<u32> = ids.iter().map(|t| *t as u32).collect();
        self.tok.decode(&ids, true).unwrap_or_default()
    }

    /// Qwen2.5's chat template with one user message, ready for the reply (apply_chat_template, add_generation_prompt).
    pub fn chat(prompt: &str) -> String {
        format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
                 <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// Sample one reply per prompt (up to 8 at once, side by side). on_text gets the first reply's text as it
    /// grows (for streaming). Stops early when stop is set. Returns each reply's text.
    pub fn generate(&mut self, prompts: &[String], max_new: &[usize], temperature: f32, stop: &AtomicBool,
                    mut on_text: Option<&mut dyn FnMut(&str) -> bool>) -> Result<Vec<String>> {
        let n = prompts.len();
        if n == 0 {
            return Ok(vec![]);
        }
        assert!(n <= N_SEQ as usize);
        let mut rng = rand::rng();
        unsafe {
            let mem = ll::llama_get_memory(self.ctx);
            ll::llama_memory_clear(mem, true);
            let mut batch = ll::llama_batch_init(N_BATCH as i32, 0, N_SEQ as i32);
            let same = prompts.iter().all(|p| p == &prompts[0]);
            let toks: Vec<Vec<i32>> = prompts.iter().map(|p| self.tokens(p)).collect();
            let mut pos = vec![0usize; n];
            // the prompts, packed into batches; one shared prompt is read once and copied to every sequence
            let feeds: Vec<usize> = if same { vec![0] } else { (0..n).collect() };
            let mut flat: Vec<(usize, i32, bool)> = vec![];
            for &s in &feeds {
                for (i, t) in toks[s].iter().enumerate() {
                    flat.push((s, *t, i + 1 == toks[s].len()));
                }
            }
            let mut logits: Vec<Vec<f32>> = vec![vec![]; n];
            for chunk in flat.chunks(N_BATCH as usize) {
                batch.n_tokens = 0;
                for (s, t, last) in chunk {
                    put(&mut batch, *t, pos[*s] as i32, *s as i32, *last);
                    pos[*s] += 1;
                }
                if ll::llama_decode(self.ctx, batch) != 0 {
                    ll::llama_batch_free(batch);
                    return Err(anyhow!("the model could not read the prompt (too long?)"));
                }
                for (i, (s, _, last)) in chunk.iter().enumerate() {
                    if *last {
                        logits[*s] = std::slice::from_raw_parts(ll::llama_get_logits_ith(self.ctx, i as i32), self.n_vocab).to_vec();
                    }
                }
            }
            if same {
                for s in 1..n {
                    ll::llama_memory_seq_cp(mem, 0, s as i32, -1, -1);
                    pos[s] = pos[0];
                    logits[s] = logits[0].clone();
                }
            }
            let mut out: Vec<Vec<i32>> = vec![vec![]; n];
            let mut alive: Vec<bool> = vec![true; n];
            let mut shown = 0usize;
            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // sample the next token of every live sequence
                batch.n_tokens = 0;
                let mut rows = vec![];
                for s in 0..n {
                    if !alive[s] {
                        continue;
                    }
                    let t = sample(&logits[s], temperature, &mut rng);
                    if ll::llama_vocab_is_eog(self.vocab, t) {
                        alive[s] = false;
                        continue;
                    }
                    out[s].push(t);
                    if out[s].len() >= max_new[s] {
                        alive[s] = false;
                        continue;
                    }
                    put(&mut batch, t, pos[s] as i32, s as i32, true);
                    pos[s] += 1;
                    rows.push(s);
                }
                if let Some(f) = on_text.as_mut() {
                    let text = self.decode(&out[0]);
                    let ready = text.trim_end_matches('\u{fffd}');
                    if ready.len() > shown {
                        if !f(&ready[shown..]) {
                            stop.store(true, Ordering::Relaxed);
                        }
                        shown = ready.len();
                    }
                }
                if rows.is_empty() {
                    break;
                }
                if ll::llama_decode(self.ctx, batch) != 0 {
                    ll::llama_batch_free(batch);
                    return Err(anyhow!("generation failed (out of memory?)"));
                }
                for (i, s) in rows.iter().enumerate() {
                    logits[*s] = std::slice::from_raw_parts(ll::llama_get_logits_ith(self.ctx, i as i32), self.n_vocab).to_vec();
                }
            }
            ll::llama_batch_free(batch);
            if let Some(f) = on_text.as_mut() {
                let text = self.decode(&out[0]);
                if text.len() > shown {
                    f(&text[shown..]);
                }
            }
            Ok(out.iter().map(|o| self.decode(o)).collect())
        }
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        unsafe {
            ll::llama_free(self.ctx);
            for (_, a) in self.adapters.drain() {
                ll::llama_adapter_lora_free(a);
            }
            ll::llama_model_free(self.model);
        }
    }
}

unsafe fn put(b: &mut ll::llama_batch, tok: i32, pos: i32, seq: i32, logits: bool) {
    let k = b.n_tokens as usize;
    *b.token.add(k) = tok;
    *b.pos.add(k) = pos;
    *b.n_seq_id.add(k) = 1;
    *(*b.seq_id.add(k)) = seq;
    *b.logits.add(k) = logits as i8;
    b.n_tokens += 1;
}

/// A token from softmax(logits / temperature).
fn sample(logits: &[f32], t: f32, rng: &mut impl Rng) -> i32 {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let w: Vec<f64> = logits.iter().map(|l| (((l - m) / t) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.random::<f64>() * total;
    for (i, x) in w.iter().enumerate() {
        r -= x;
        if r <= 0.0 {
            return i as i32;
        }
    }
    (w.len() - 1) as i32
}
