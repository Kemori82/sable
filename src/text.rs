//! Text handling ported from the Python version, kept to the same rules so scores and corrections match:
//! the detector's cleaning and segments (detector.py), the humanizer's passages (generator.py), and the
//! corrector's sentences, repeats, diffs and grammar-fix checks (corrector.py).
use crate::difflib::{SequenceMatcher, Tag};
use fancy_regex::Regex as FRegex;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

// ---- Python string helpers ----

/// str.isspace(): Unicode white space plus the separators \x1c-\x1f that Python also counts.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// str.split() with no arguments.
pub fn py_split(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|w| !w.is_empty()).collect()
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

pub fn n_words(s: &str) -> usize {
    py_split(s).len()
}

/// Python's round(): halves go to the even neighbour.
pub fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

/// re.split for a pattern without groups.
fn fsplit<'a>(re: &FRegex, s: &'a str) -> Vec<&'a str> {
    let mut out = vec![];
    let mut last = 0;
    for m in re.find_iter(s).flatten() {
        out.push(&s[last..m.start()]);
        last = m.end();
    }
    out.push(&s[last..]);
    out
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

// ---- detector cleaning (detector.normalize) ----

static REF_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s(References|Works Cited|Reference List|Bibliography)\s").unwrap());
static CITATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\((?:[^()]*?\b(?:19|20)\d{2}[a-z]?\b[^()]*?)\)").unwrap());
static URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://\S+|www\.\S+").unwrap());
static PLACEHOLDERS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (Regex::new(r"\b(?:Generic_Name|STUDENT_NAME|TEACHER_NAME|PROPER_NAME|Generic_Teacher)\b").unwrap(), "Alex"),
        (Regex::new(r"\b(?:Generic_City|LOCATION_NAME|Generic_Location|Generic_State)\b").unwrap(), "Springfield"),
        (Regex::new(r"\b(?:Generic_School|SCHOOL_NAME)\b").unwrap(), "Lincoln School"),
        (Regex::new(r"\bGeneric_\w+\b").unwrap(), "Alex"),
    ]
});
static LEAD_HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(Introduction|Abstract)\b[:.]?\s+").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\x1c-\x1f]+").unwrap());
static SPACE_PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\x1c-\x1f]+([.,;:])").unwrap());

fn quote(c: char) -> Option<char> {
    Some(match c {
        '\u{2018}' | '\u{2019}' => '\'',
        '\u{201c}' | '\u{201d}' => '"',
        '\u{2013}' | '\u{2014}' => '-',
        _ => return None,
    })
}

fn zero_width(c: char) -> bool {
    matches!(c, '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}')
}

/// Cyrillic / Greek letters that look Latin.
pub fn confusable(c: char) -> Option<char> {
    Some(match c {
        '\u{0430}' => 'a', '\u{0435}' => 'e', '\u{0456}' => 'i', '\u{043e}' => 'o', '\u{0441}' => 'c',
        '\u{0440}' => 'p', '\u{0443}' => 'y', '\u{0445}' => 'x', '\u{0455}' => 's', '\u{0458}' => 'j',
        '\u{04bb}' => 'h', '\u{0410}' => 'A', '\u{0412}' => 'B', '\u{0415}' => 'E', '\u{041a}' => 'K',
        '\u{041c}' => 'M', '\u{041d}' => 'H', '\u{041e}' => 'O', '\u{0420}' => 'P', '\u{0421}' => 'C',
        '\u{0422}' => 'T', '\u{0425}' => 'X', '\u{0406}' => 'I', '\u{0391}' => 'A', '\u{0392}' => 'B',
        '\u{0395}' => 'E', '\u{0396}' => 'Z', '\u{0397}' => 'H', '\u{0399}' => 'I', '\u{039a}' => 'K',
        '\u{039c}' => 'M', '\u{039d}' => 'N', '\u{039f}' => 'O', '\u{03a1}' => 'P', '\u{03a4}' => 'T',
        '\u{03a7}' => 'X', '\u{03bf}' => 'o',
        _ => return None,
    })
}

pub fn has_confusables(s: &str) -> bool {
    s.chars().any(|c| confusable(c).is_some())
}

/// The same cleaning the detector was trained on.
pub fn normalize(text: &str) -> String {
    let t: String = text.nfkc().map(|c| quote(c).unwrap_or(c)).collect();
    let mut t: String = t.chars().filter(|c| !zero_width(*c)).map(|c| confusable(c).unwrap_or(c)).collect();
    if let Some(m) = REF_HEADING.find_iter(&t).last() {
        // trailing reference list: only if the heading is in the second half (Python compares code points)
        if char_len(&t[..m.start()]) as f64 > 0.5 * char_len(&t) as f64 {
            t.truncate(m.start());
        }
    }
    for (re, repl) in PLACEHOLDERS.iter() {
        t = re.replace_all(&t, *repl).into_owned();
    }
    let t = URL.replace_all(&t, "");
    let t = CITATION.replace_all(&t, "");
    let t = SPACES.replace_all(&t, " ");
    let t = LEAD_HEADING.replace(py_strip(&t), "").into_owned();
    SPACE_PUNCT.replace_all(&t, "$1").into_owned()
}

pub const ENGLISH_WORDS: &[&str] = &["a", "all", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "for",
    "from", "has", "have", "her", "his", "in", "is", "it", "more", "not", "of", "on", "one", "or", "that", "the", "their",
    "there", "they", "this", "to", "was", "we", "were", "which", "with", "would", "you"];

pub fn english_share(words: &[&str]) -> f64 {
    let set: HashSet<&str> = ENGLISH_WORDS.iter().copied().collect();
    let n = words
        .iter()
        .filter(|w| set.contains(w.trim_matches(|c| ".,;:!?\"'()".contains(c)).to_lowercase().as_str()))
        .count();
    n as f64 / words.len().max(1) as f64
}

// ---- detector segments (segments.py) ----

static SENTENCE_END: LazyLock<FRegex> =
    LazyLock::new(|| FRegex::new(r#"(?<=[.!?])["')\]]?\s+(?=[A-Z0-9"'(\[])"#).unwrap());

/// Sentences packed into segments of about 60 words (at most 120), a short tail merged into the last one.
/// Returned as byte ranges into text (char ranges for the page: see char_span).
pub fn segment(text: &str) -> Vec<(usize, usize)> {
    let mut spans = vec![];
    let mut cursor = 0;
    for piece in fsplit(&SENTENCE_END, text) {
        let s = py_strip(piece);
        if s.is_empty() {
            continue;
        }
        if let Some(off) = text[cursor..].find(s) {
            let start = cursor + off;
            spans.push((start, start + s.len()));
            cursor = start + s.len();
        }
    }
    let mut out: Vec<(usize, usize)> = vec![];
    let (mut cur, mut words): (Option<(usize, usize)>, usize) = (None, 0);
    for (start, end) in spans {
        let w = n_words(&text[start..end]);
        if cur.is_some() && (words >= 60 || words + w > 120) {
            out.push(cur.unwrap());
            cur = None;
            words = 0;
        }
        cur = Some(match cur {
            Some((s, _)) => (s, end),
            None => (start, end),
        });
        words += w;
    }
    if let Some(c) = cur {
        if !out.is_empty() && words < 25 {
            out.last_mut().unwrap().1 = c.1;
        } else {
            out.push(c);
        }
    }
    out
}

/// A byte range as code-point offsets, as the Python version returned them to the page.
pub fn char_span(text: &str, (a, b): (usize, usize)) -> (usize, usize) {
    let ca = char_len(&text[..a]);
    (ca, ca + char_len(&text[a..b]))
}

// ---- humanizer passages (generator.chunks) ----

static PARA_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n[\s\x1c-\x1f]*\n").unwrap());
static CHUNK_SENT: LazyLock<FRegex> = LazyLock::new(|| FRegex::new(r#"(?<=[.!?])\s+(?=[A-Z"'(])"#).unwrap());

/// Paragraphs, each split into passages of at most max_sents sentences (the humanizer trained on 3-5).
pub fn chunks(text: &str, max_sents: usize) -> Vec<Vec<String>> {
    let mut out = vec![];
    for para in PARA_SPLIT.split(text).map(py_strip).filter(|p| !p.is_empty()) {
        let s: Vec<&str> = fsplit(&CHUNK_SENT, para).into_iter().filter(|x| !x.is_empty()).collect();
        let n = py_round(s.len() as f64 / max_sents as f64).max(1) as usize;
        let size = s.len().div_ceil(n).max(1);
        out.push(s.chunks(size).map(|c| c.join(" ")).collect());
    }
    out
}

// ---- corrector ----

pub fn paragraphs(text: &str) -> Vec<String> {
    PARA_SPLIT.split(text).map(py_strip).filter(|p| !p.is_empty()).map(String::from).collect()
}

static SENT: LazyLock<FRegex> = LazyLock::new(|| FRegex::new(r#"(?<=[.!?])\s+(?=[A-Z"'(*\[])"#).unwrap());

pub fn sentences(text: &str) -> Vec<String> {
    fsplit(&SENT, py_strip(text)).into_iter().filter(|s| !s.is_empty()).map(String::from).collect()
}

static REPEAT: LazyLock<FRegex> = LazyLock::new(|| FRegex::new(r"((?:\S+[ \t]+){1,8}?)\1{2,}").unwrap());
static NONWORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\W+").unwrap());

/// A run of 1-8 words repeated 3+ times in a row cut to one; a sentence seen before (same words) dropped.
pub fn unrepeat(text: &str) -> String {
    let t = format!("{text} ");
    let t = REPEAT.replace_all(&t, "$1");
    let t = py_strip(&t);
    let mut out = vec![];
    let mut seen = HashSet::new();
    for para in paragraphs(t) {
        let mut keep = vec![];
        for s in sentences(&para) {
            let key = NONWORD.replace_all(&s.to_lowercase(), " ").trim_matches(is_py_space).to_string();
            if !key.is_empty() && seen.insert(key) {
                keep.push(s);
            }
        }
        if !keep.is_empty() {
            out.push(keep.join(" "));
        }
    }
    out.join("\n\n")
}

fn words(s: &str) -> Vec<String> {
    py_split(s).into_iter().map(String::from).collect()
}

/// b with a's spacing wherever the two differ only in spaces (small T5 models write "Name ]." for "Name].").
pub fn respace(a: &str, b: &str) -> String {
    let (wa, wb) = (words(a), words(b));
    let mut out: Vec<&str> = vec![];
    for (tag, i1, i2, j1, j2) in SequenceMatcher::new(&wa, &wb, false).get_opcodes() {
        let same = wa[i1..i2].concat() == wb[j1..j2].concat();
        if tag == Tag::Equal || same {
            out.extend(wa[i1..i2].iter().map(|s| s.as_str()));
        } else {
            out.extend(wb[j1..j2].iter().map(|s| s.as_str()));
        }
    }
    out.join(" ")
}

pub fn small_edit(a: &str, b: &str) -> bool {
    let (wa, wb) = (words(a), words(b));
    SequenceMatcher::new(&wa, &wb, true).ratio() >= 0.8 && wb.len() <= wa.len() + 4
}

static GRAMMAR_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    "a an the this that these those some any each every no all both either neither another such
in on at of for to with about from by into onto over under after before during since until through between among
within without against toward towards upon off out up down as than like per via around across along behind beyond
and or but nor so yet if because although though while when where whether which who whom whose what why how
is are was were be been being am has have had having do does did done will would shall should can could may might
must it its they them their there we us our you your he him his she her i me my one ones not more most much many
less few also too only just even then"
        .split_whitespace()
        .collect()
});
const PRONOUNS: &[&str] = &["it", "they", "them", "we", "us", "you", "he", "him", "she", "her", "i", "me", "this", "these", "those"];
const SAFE_PUNCT: &str = ",.;?!'";

static EDGE_NONWORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\W+|\W+$").unwrap());
static WORDCHAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w").unwrap());
static NOT_WORDCHAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\W").unwrap());

pub fn bare(w: &str) -> String {
    EDGE_NONWORD.replace_all(w, "").to_lowercase()
}

/// Is changing words a into b a grammar-sized fix? (corrector.safe_change)
pub fn safe_change(a: &[String], b: &[String], before: &str, after: &str) -> bool {
    let punct = |ws: &[String]| -> HashSet<char> { ws.iter().flat_map(|w| WORDCHAR.replace_all(w, "").chars().collect::<Vec<_>>()).collect() };
    let (pa, pb) = (punct(a), punct(b));
    if pb.difference(&pa).any(|c| !SAFE_PUNCT.contains(*c)) {
        return false;
    }
    let ba: Vec<String> = a.iter().map(|w| bare(w)).filter(|w| !w.is_empty()).collect();
    let bb: Vec<String> = b.iter().map(|w| bare(w)).filter(|w| !w.is_empty()).collect();
    if ba == bb {
        // only punctuation or capitals changed: capitals only at the start of the sentence
        let strip = |ws: &[String]| -> Vec<String> { ws.iter().map(|w| NOT_WORDCHAR.replace_all(w, "").into_owned()).collect() };
        return strip(a) == strip(b) || before.is_empty();
    }
    let pre3 = |x: &str| -> String { x.chars().take(3).collect() };
    let related = |x: &str, ys: &[String]| ys.iter().any(|y| pre3(x) == pre3(y));
    for x in &ba {
        if bb.contains(x) || related(x, &bb)
            || (GRAMMAR_WORDS.contains(x.as_str()) && !bb.is_empty() && bb.iter().all(|y| GRAMMAR_WORDS.contains(y.as_str())))
        {
            continue;
        }
        if PRONOUNS.contains(&x.as_str()) || ["of", "a", "an"].contains(&x.as_str()) {
            return false;
        }
        if !(x == "the" || x == "that" || *x == bare(before) || *x == bare(after)) {
            return false;
        }
    }
    for y in &bb {
        if !(ba.contains(y) || related(y, &ba) || GRAMMAR_WORDS.contains(y.as_str())) {
            return false;
        }
    }
    true
}

/// b with each change that isn't a grammar-sized fix put back to a (Corrector.safe).
pub fn safe(a: &str, b: &str) -> String {
    let (wa, wb) = (words(a), words(b));
    let mut out: Vec<&str> = vec![];
    for (tag, i1, i2, j1, j2) in SequenceMatcher::new(&wa, &wb, false).get_opcodes() {
        let before = if i1 > 0 { wa[i1 - 1].as_str() } else { "" };
        let after = if i2 < wa.len() { wa[i2].as_str() } else { "" };
        let keep = tag == Tag::Equal || safe_change(&wa[i1..i2], &wb[j1..j2], before, after);
        let src = if keep { &wb[j1..j2] } else { &wa[i1..i2] };
        out.extend(src.iter().map(|s| s.as_str()));
    }
    out.join(" ")
}

static TOKEN_WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\s\x1c-\x1f]+[\s\x1c-\x1f]*").unwrap());

/// Word-level diff: [{op: eq|del|ins, text}], whitespace kept with the words.
pub fn diff(a: &str, b: &str) -> Vec<serde_json::Value> {
    let ta: Vec<&str> = TOKEN_WS.find_iter(a).map(|m| m.as_str()).collect();
    let tb: Vec<&str> = TOKEN_WS.find_iter(b).map(|m| m.as_str()).collect();
    let sa: Vec<String> = ta.iter().map(|t| py_strip(t).to_string()).collect();
    let sb: Vec<String> = tb.iter().map(|t| py_strip(t).to_string()).collect();
    let mut out = vec![];
    for (tag, i1, i2, j1, j2) in SequenceMatcher::new(&sa, &sb, false).get_opcodes() {
        if tag == Tag::Equal {
            out.push(serde_json::json!({"op": "eq", "text": tb[j1..j2].concat()}));
        } else {
            if i2 > i1 {
                out.push(serde_json::json!({"op": "del", "text": ta[i1..i2].concat()}));
            }
            if j2 > j1 {
                out.push(serde_json::json!({"op": "ins", "text": tb[j1..j2].concat()}));
            }
        }
    }
    out
}
