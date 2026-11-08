//! ONNX Runtime sessions for the encoder models (detector, meaning checker, grammar fixer): DirectML on any
//! DirectX 12 graphics card (NVIDIA, AMD, Intel), otherwise the CPU. Weights are half precision.
use anyhow::{anyhow, Result};
use ort::{ep, session::Session, value::Tensor};
use std::path::Path;
use std::sync::OnceLock;

static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
static GPU: OnceLock<bool> = OnceLock::new();

fn env() -> Result<&'static ort::environment::Environment> {
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let e = ort::init().with_name("sable").build().map_err(|e| anyhow!("{e}"))?;
    Ok(ENV.get_or_init(|| e))
}

/// Whether the encoder models run on the graphics card (decided by the first model that loads).
pub fn on_gpu() -> bool {
    *GPU.get().unwrap_or(&false)
}

/// On the CPU whatever the graphics card: for the grammar fixer, whose many tiny steps run faster there than
/// with a copy to the graphics card and back at every step.
pub fn cpu_session(path: &Path) -> Result<Session> {
    if std::env::var("SABLE_GEC_GPU").is_ok() {
        return session(path);
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    Ok(Session::builder(env()?)?.with_intra_threads(threads).map_err(|e| anyhow!("{e}"))?.commit_from_file(path)?)
}

pub fn session(path: &Path) -> Result<Session> {
    let env = env()?;
    let cpu = || -> Result<Session> {
        Ok(Session::builder(env)?.with_intra_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(path)?)
    };
    if std::env::var("SABLE_CPU").is_ok() || GPU.get() == Some(&false) {
        GPU.get_or_init(|| false);
        return cpu();
    }
    let gpu = (|| -> Result<Session> {
        Ok(Session::builder(env)?
            .with_memory_pattern(false).map_err(|e| anyhow!("{e}"))?
            .with_parallel_execution(false).map_err(|e| anyhow!("{e}"))?
            .with_execution_providers([ep::DirectML::default().build().error_on_failure()]).map_err(|e| anyhow!("{e}"))?
            .commit_from_file(path)?)
    })();
    match gpu {
        Ok(s) => {
            GPU.get_or_init(|| true);
            Ok(s)
        }
        Err(_) => {
            GPU.get_or_init(|| false);
            cpu()
        }
    }
}

/// [rows, cols] int64 tensor
pub fn i64_tensor(rows: usize, cols: usize, data: Vec<i64>) -> Result<Tensor<i64>> {
    Ok(Tensor::from_array((vec![rows as i64, cols as i64], data))?)
}

/// Rounded up to one of a few sizes: DirectML prepares its work per input shape, so reusing a handful of
/// shapes is much faster than a new one for every batch. Padding is masked out, so results don't change.
pub fn bucket(n: usize) -> usize {
    if n <= 16 { n.next_power_of_two().max(1) } else { n.div_ceil(32) * 32 }
}

/// Token ids padded to a bucketed length, and padded with dummy rows to a bucketed count: (ids, mask, types)
/// flattened, the row length, and the row count (only the first encs.len() rows are real).
pub fn pad_bucketed(encs: &[tokenizers::Encoding], pad_id: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>, usize, usize) {
    let longest = encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
    let len = if on_gpu() { bucket(longest) } else { longest };
    let rows = if on_gpu() { bucket(encs.len()) } else { encs.len() };
    let (mut ids, mut mask, mut types) = (vec![pad_id; rows * len], vec![0i64; rows * len], vec![0i64; rows * len]);
    for (i, e) in encs.iter().enumerate() {
        for (j, (t, ty)) in e.get_ids().iter().zip(e.get_type_ids()).enumerate() {
            ids[i * len + j] = *t as i64;
            mask[i * len + j] = 1;
            types[i * len + j] = *ty as i64;
        }
    }
    for i in encs.len()..rows {
        mask[i * len] = 1; // a dummy row needs one visible token
    }
    (ids, mask, types, len, rows)
}

/// Token ids padded to the longest row: (ids, mask, [types]) flattened, and the row length.
pub fn pad(encs: &[tokenizers::Encoding], pad_id: i64) -> (Vec<i64>, Vec<i64>, Vec<i64>, usize) {
    let len = encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
    let n = encs.len();
    let (mut ids, mut mask, mut types) = (vec![pad_id; n * len], vec![0i64; n * len], vec![0i64; n * len]);
    for (i, e) in encs.iter().enumerate() {
        for (j, (t, ty)) in e.get_ids().iter().zip(e.get_type_ids()).enumerate() {
            ids[i * len + j] = *t as i64;
            mask[i * len + j] = 1;
            types[i * len + j] = *ty as i64;
        }
    }
    (ids, mask, types, len)
}

pub fn tokenizer(path: &Path, max_len: usize) -> Result<tokenizers::Tokenizer> {
    let mut t = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("{e}"))?;
    t.with_truncation(Some(tokenizers::TruncationParams { max_length: max_len, ..Default::default() }))
        .map_err(|e| anyhow!("{e}"))?;
    t.with_padding(None);
    Ok(t)
}

/// Time spent per model, for SABLE_DEBUG profiling: (name, nanoseconds, calls).
pub static TIMES: std::sync::Mutex<Vec<(&'static str, u128, u64)>> = std::sync::Mutex::new(Vec::new());

pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let r = f();
    let ns = t.elapsed().as_nanos();
    let mut v = TIMES.lock().unwrap();
    match v.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += ns;
            e.2 += 1;
        }
        None => v.push((name, ns, 1)),
    }
    r
}

pub fn report() {
    for (n, ns, c) in TIMES.lock().unwrap().iter() {
        eprintln!("  {n}: {:.1}s in {c} calls", *ns as f64 / 1e9);
    }
}
