//! Corrector: takes the original text and its humanized version, puts back meaning that was lost, and fixes
//! grammar without undoing the humanizing (corrector.py, gentle mode: the only one the app used).
//!
//!   0. Repeats   a phrase repeated over and over is cut to one, and repeated sentences are dropped.
//!   1. Meaning   the NLI model checks each paragraph both ways: an original sentence the humanized paragraph
//!                doesn't support is missing; a humanized sentence the original contradicts or doesn't support
//!                at all is wrong.
//!   2. Repair    a wrong sentence is taken out, a missing point goes back in as one new sentence humanized on
//!                its own. Each change is kept only if it lowers the problem count and keeps the paragraph under
//!                the detector limit.
//!   3. Grammar   T5 fixes each sentence; a fix is kept only if it is small and the paragraph's detector score
//!                doesn't rise past the line. With safe_check (the Tuned option), changes that aren't
//!                grammar-sized are put back first.
//!   4. Whole-text gate: paragraphs that raised the whole text's score most are put back until it is under.
use crate::{gec::Gec, nli::Nli, text};
use anyhow::Result;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;

const MISSING: f32 = 0.5;
const CONTRA: f32 = 0.5;
const MADE_UP: f32 = 0.05;
const LINE: f64 = 0.35;

/// One sentence, no 3-word run repeated (the repair test's "clean single sentence").
fn tidy(x: &str) -> bool {
    let w: Vec<String> = text::py_split(x).iter().map(|s| s.to_lowercase()).collect();
    let grams: Vec<&[String]> = w.windows(3).collect();
    let uniq: std::collections::HashSet<&[String]> = grams.iter().copied().collect();
    let t = x.trim();
    let ends = t.char_indices().filter(|(i, ch)| ".!?".contains(*ch)
        && t[i + ch.len_utf8()..].chars().next().map(|n| n.is_whitespace()).unwrap_or(true)).count();
    ends <= 1 && uniq.len() == grams.len()
}

pub struct Corrector {
    pub nli: Nli,
    pub gec: Gec,
    pub safe_check: bool,
}

struct Check {
    orig: Vec<String>,
    hum: Vec<String>,
    missing: Vec<usize>,
    wrong: Vec<usize>,
}

impl Check {
    fn problems(&self) -> usize {
        self.missing.len() + self.wrong.len()
    }
}

/// The callbacks the corrector uses: the humanizer, the repair model (if on) and the detector.
pub struct Tools<'a> {
    pub rewrite: &'a mut dyn FnMut(&str) -> Result<Vec<String>>,
    /// (original paragraph, original sentence, humanized paragraph with the sentence to fix in [[ ]]) -> candidate fixes
    pub fix: Option<&'a mut dyn FnMut(&str, &str, &str) -> Result<Vec<String>>>,
    pub detect: &'a dyn Fn(&[String]) -> Result<Vec<f64>>,
    pub score_doc: &'a dyn Fn(&str) -> Result<f64>,
    pub progress: &'a dyn Fn(&str),
}

/// Detector answers remembered for the length of one correction (the same paragraph is scored many times).
struct Cached<'a> {
    detect: &'a dyn Fn(&[String]) -> Result<Vec<f64>>,
    score_doc: &'a dyn Fn(&str) -> Result<f64>,
    passage: RefCell<HashMap<String, f64>>,
    doc: RefCell<HashMap<String, f64>>,
}

impl Cached<'_> {
    fn detect(&self, texts: &[String]) -> Result<Vec<f64>> {
        let todo: Vec<String> = {
            let c = self.passage.borrow();
            let mut seen = std::collections::HashSet::new();
            texts.iter().filter(|t| !c.contains_key(*t) && seen.insert(t.as_str())).cloned().collect()
        };
        if !todo.is_empty() {
            let r = (self.detect)(&todo)?;
            let mut c = self.passage.borrow_mut();
            for (t, p) in todo.into_iter().zip(r) {
                c.insert(t, p);
            }
        }
        let c = self.passage.borrow();
        Ok(texts.iter().map(|t| c[t]).collect())
    }

    fn one(&self, t: &str) -> Result<f64> {
        Ok(self.detect(&[t.to_string()])?[0])
    }

    fn doc(&self, t: &str) -> Result<f64> {
        if let Some(v) = self.doc.borrow().get(t) {
            return Ok(*v);
        }
        let v = (self.score_doc)(t)?;
        self.doc.borrow_mut().insert(t.to_string(), v);
        Ok(v)
    }
}

impl Corrector {
    fn check(&self, orig: &str, hum: &str) -> Result<Check> {
        let (o, h) = (text::sentences(orig), text::sentences(hum));
        let mut prem = vec![hum.to_string(); o.len()];
        prem.extend(std::iter::repeat_n(orig.to_string(), h.len()));
        let hyp: Vec<String> = o.iter().chain(&h).cloned().collect();
        let p = self.nli.probs(&prem, &hyp)?;
        let missing = (0..o.len()).filter(|&i| p[i][self.nli.ent] < MISSING).collect();
        let wrong = (0..h.len())
            .filter(|&j| p[o.len() + j][self.nli.con] > CONTRA || p[o.len() + j][self.nli.ent] < MADE_UP)
            .collect();
        Ok(Check { orig: o, hum: h, missing, wrong })
    }

    /// For each humanized sentence, the original sentence it came from: the one that supports it best, or that it
    /// supports best (NLI, both ways), plus how many content words they share.
    fn align(&self, orig: &[String], hum: &[String]) -> Result<Vec<usize>> {
        let (mut prem, mut hyp) = (vec![], vec![]);
        for h in hum {
            for o in orig {
                prem.push(o.clone());
                hyp.push(h.clone());
                prem.push(h.clone());
                hyp.push(o.clone());
            }
        }
        let p = self.nli.probs(&prem, &hyp)?;
        let words = |s: &str| -> std::collections::HashSet<String> {
            text::py_split(s).iter().map(|w| text::bare(w)).filter(|w| w.len() > 3).collect()
        };
        Ok((0..hum.len()).map(|j| {
            let wh = words(&hum[j]);
            let score = |i: usize| {
                let k = 2 * (j * orig.len() + i);
                let wo = words(&orig[i]);
                let overlap = wh.intersection(&wo).count() as f32 / wh.union(&wo).count().max(1) as f32;
                p[k][self.nli.ent].max(p[k + 1][self.nli.ent]) + 0.5 * overlap
            };
            (0..orig.len()).max_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap()).unwrap_or(0)
        }).collect())
    }

    /// Fix sentences in place (the repair model's fixes and Writer voice's rewrites of the original sentence, best
    /// first): each humanized sentence that is wrong, or that came from an original sentence the paragraph lost.
    /// A fix is kept only if the original sentence and the fix support each other, the paragraph's problems go
    /// down and it stays under the detector limit; a clean single sentence is preferred.
    fn fix_in_place(&self, o: &str, c: &Check, fix: &mut dyn FnMut(&str, &str, &str) -> Result<Vec<String>>,
                    rewrite: &mut dyn FnMut(&str) -> Result<Vec<String>>, det: &Cached, limit: f64)
                    -> Result<(Vec<String>, usize)> {
        let (mut s, orig, mut probs) = (c.hum.clone(), &c.orig, c.problems());
        if orig.is_empty() || s.is_empty() {
            return Ok((s, 0));
        }
        let align = self.align(orig, &s)?;
        let mut done = std::collections::HashSet::new();
        let mut repaired = 0;
        for j in 0..s.len() {
            let i = align[j];
            if !(c.wrong.contains(&j) || c.missing.contains(&i)) || !done.insert(i) {
                continue;
            }
            let marked: Vec<String> = s.iter().enumerate().map(|(k, x)| if k == j { format!("[[{x}]]") } else { x.clone() }).collect();
            let mut cands: Vec<String> = fix(o, &orig[i], &marked.join(" "))?;
            cands.extend(rewrite(&orig[i])?);
            let mut seen = std::collections::HashSet::new();
            let cands: Vec<String> = cands.into_iter()
                .map(|x| text::py_strip(&x.replace("[[", "").replace("]]", "")).trim_matches('"').to_string())
                .filter(|x| !x.is_empty() && *x != s[j] && seen.insert(x.clone())).collect();
            if cands.is_empty() {
                continue;
            }
            let n = cands.len();
            let fwd = self.nli.probs(&vec![orig[i].clone(); n], &cands)?;
            let back = self.nli.probs(&cands, &vec![orig[i].clone(); n])?;
            let trials: Vec<Vec<String>> = cands.iter().map(|x| {
                let mut t = s.clone();
                t[j] = x.clone();
                t
            }).collect();
            let ai = det.detect(&trials.iter().map(|t| t.join(" ")).collect::<Vec<_>>())?;
            let key = |k: usize| {
                let ent = fwd[k][self.nli.ent].min(back[k][self.nli.ent]);
                let con = fwd[k][self.nli.con].max(back[k][self.nli.con]);
                (ent >= 0.5 && con < 0.5, ai[k] <= limit, tidy(&cands[k]), ent as f64 - 0.5 * (ai[k] - 0.35).max(0.0))
            };
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|a, b| key(*b).partial_cmp(&key(*a)).unwrap());
            for k in order.into_iter().take(3) {
                if !key(k).0 || !key(k).1 {
                    break;
                }
                let p = self.check(o, &trials[k].join(" "))?.problems();
                if p < probs {
                    s = trials[k].clone();
                    probs = p;
                    repaired += 1;
                    break;
                }
            }
        }
        Ok((s, repaired))
    }

    fn repair<'f>(&self, o: &str, h: &str, c: &Check, rewrite: &mut dyn FnMut(&str) -> Result<Vec<String>>,
              fix: Option<&mut (dyn FnMut(&str, &str, &str) -> Result<Vec<String>> + 'f)>,
              det: &Cached) -> Result<(String, usize, usize, usize)> {
        let limit = LINE.max(det.one(h)? + 0.05);
        let mut repaired = 0;
        let fixed;
        let c = match fix {
            Some(f) => {
                let (s, n) = self.fix_in_place(o, c, f, rewrite, det, limit)?;
                repaired = n;
                fixed = self.check(o, &s.join(" "))?;
                &fixed
            }
            None => c,
        };
        let (mut s, orig, mut probs) = (c.hum.clone(), &c.orig, c.problems());
        let (mut restored, mut removed) = (0, 0);
        let mut wrong = c.wrong.clone();
        wrong.sort_unstable_by(|a, b| b.cmp(a));
        for j in wrong {
            let mut trial = s.clone();
            trial.remove(j);
            if trial.is_empty() {
                continue;
            }
            let joined = trial.join(" ");
            let p = self.check(o, &joined)?.problems();
            if p < probs && det.one(&joined)? <= limit {
                s = trial;
                probs = p;
                removed += 1;
            }
        }
        for i in self.check(o, &s.join(" "))?.missing {
            let mut cands: Vec<String> = rewrite(&orig[i])?.into_iter().filter(|x| !text::py_strip(x).is_empty()).collect();
            cands.push(orig[i].clone());
            let keeps: Vec<f32> = self.nli.probs(&cands, &vec![orig[i].clone(); cands.len()])?
                .iter().map(|p| p[self.nli.ent]).collect();
            let pos = (text::py_round(i as f64 / orig.len().max(1) as f64 * s.len() as f64).max(0) as usize).min(s.len());
            let trials: Vec<Vec<String>> = cands.iter().map(|x| {
                let mut t = s[..pos].to_vec();
                t.push(x.clone());
                t.extend_from_slice(&s[pos..]);
                t
            }).collect();
            let ai = det.detect(&trials.iter().map(|t| t.join(" ")).collect::<Vec<_>>())?;
            let mut order: Vec<usize> = (0..cands.len()).collect();
            order.sort_by(|a, b| ((keeps[*a] < MISSING), ai[*a]).partial_cmp(&((keeps[*b] < MISSING), ai[*b])).unwrap());
            for k in order.into_iter().take(3) {
                let joined = trials[k].join(" ");
                let p = self.check(o, &joined)?.problems();
                if p < probs && ai[k] <= limit {
                    s = trials[k].clone();
                    probs = p;
                    restored += 1;
                    break;
                }
            }
        }
        Ok((s.join(" "), restored, removed, repaired))
    }

    pub fn correct(&self, original: &str, humanized: &str, tools: Tools) -> Result<Value> {
        let det = Cached { detect: tools.detect, score_doc: tools.score_doc, passage: Default::default(), doc: Default::default() };
        let cleaned = text::unrepeat(humanized);
        let (mut op, mut hp) = (text::paragraphs(original), text::paragraphs(&cleaned));
        if op.len() != hp.len() {
            op = vec![op.join(" ")];
            hp = vec![hp.join(" ")];
        }
        let (mut restored, mut removed, mut grammar, mut undone, mut repaired) = (0, 0, 0, 0, 0);
        let mut fix = tools.fix;
        let repeats_cut = text::n_words(humanized) as i64 - text::n_words(&cleaned) as i64;
        let mut out = vec![];
        for (k, (o, h)) in op.iter().zip(&hp).enumerate() {
            (tools.progress)(&format!("meaning {}/{}", k + 1, op.len()));
            let c = self.check(o, h)?;
            let mut h = h.clone();
            if c.problems() > 0 {
                let (h2, r, m, f) = self.repair(o, &h, &c, tools.rewrite, fix.as_deref_mut(), &det)?;
                restored += r;
                removed += m;
                repaired += f;
                h = h2;
            }
            out.push(h);
        }
        (tools.progress)("grammar");
        let mut fixes = vec![];
        for h in out.iter_mut() {
            let mut s = text::sentences(h);
            let fixed = if s.is_empty() { vec![] } else { self.gec.fix(&s)? };
            let mut base = det.one(h)?;
            for (i, f) in fixed.into_iter().enumerate() {
                let f = if self.safe_check { text::safe(&s[i], &f) } else { f };
                if f == s[i] || !text::small_edit(&s[i], &f) {
                    continue;
                }
                let mut trial = s.clone();
                trial[i] = f.clone();
                let p = det.one(&trial.join(" "))?;
                if p <= base + 0.03 || p < 0.15 {
                    fixes.push((s[i].clone(), f));
                    s = trial;
                    base = p;
                    grammar += 1;
                }
            }
            *h = s.join(" ");
        }
        // whole-text gate: undo the paragraphs that raised the score most until it's back under
        let limit = LINE.max(det.doc(&cleaned)?);
        let mut gain = vec![];
        for i in 0..out.len() {
            gain.push(det.one(&hp[i])? - det.one(&out[i])?);
        }
        let mut order: Vec<usize> = (0..out.len()).collect();
        order.sort_by(|a, b| gain[*a].partial_cmp(&gain[*b]).unwrap());
        let mut order = std::collections::VecDeque::from(order);
        while !order.is_empty() && det.doc(&out.join("\n\n"))? > limit {
            let i = order.pop_front().unwrap();
            if out[i] != hp[i] {
                out[i] = hp[i].clone();
                undone += 1;
            }
        }
        let text = out.join("\n\n");
        let mut notes = json!({"restored": restored, "removed": removed, "grammar": grammar, "paragraphs_rewritten": 0,
                               "repeats_cut": repeats_cut, "repaired": repaired});
        if undone > 0 {
            notes["undone"] = json!(undone);
        }
        self.nli.clear();
        Ok(json!({"text": text, "diff": text::diff(humanized, &text), "notes": notes,
                  "grammar_fixes": fixes.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()}))
    }
}
