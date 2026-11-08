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

/// The callbacks the corrector uses: the humanizer and the detector.
pub struct Tools<'a> {
    pub rewrite: &'a mut dyn FnMut(&str) -> Result<Vec<String>>,
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

    fn repair(&self, o: &str, h: &str, c: &Check, rewrite: &mut dyn FnMut(&str) -> Result<Vec<String>>,
              det: &Cached) -> Result<(String, usize, usize)> {
        let (mut s, orig, mut probs) = (c.hum.clone(), &c.orig, c.problems());
        let limit = LINE.max(det.one(h)? + 0.05);
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
        Ok((s.join(" "), restored, removed))
    }

    pub fn correct(&self, original: &str, humanized: &str, tools: Tools) -> Result<Value> {
        let det = Cached { detect: tools.detect, score_doc: tools.score_doc, passage: Default::default(), doc: Default::default() };
        let cleaned = text::unrepeat(humanized);
        let (mut op, mut hp) = (text::paragraphs(original), text::paragraphs(&cleaned));
        if op.len() != hp.len() {
            op = vec![op.join(" ")];
            hp = vec![hp.join(" ")];
        }
        let (mut restored, mut removed, mut grammar, mut undone) = (0, 0, 0, 0);
        let repeats_cut = text::n_words(humanized) as i64 - text::n_words(&cleaned) as i64;
        let mut out = vec![];
        for (k, (o, h)) in op.iter().zip(&hp).enumerate() {
            (tools.progress)(&format!("meaning {}/{}", k + 1, op.len()));
            let c = self.check(o, h)?;
            let mut h = h.clone();
            if c.problems() > 0 {
                let (h2, r, m) = self.repair(o, &h, &c, tools.rewrite, &det)?;
                restored += r;
                removed += m;
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
                               "repeats_cut": repeats_cut});
        if undone > 0 {
            notes["undone"] = json!(undone);
        }
        self.nli.clear();
        Ok(json!({"text": text, "diff": text::diff(humanized, &text), "notes": notes,
                  "grammar_fixes": fixes.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()}))
    }
}
