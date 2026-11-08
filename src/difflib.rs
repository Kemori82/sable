//! Python's difflib.SequenceMatcher (isjunk=None), ported line for line so the corrector's diffs, "small edit"
//! ratios and safety-check opcodes come out exactly as in the Python version.
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Equal,
    Replace,
    Delete,
    Insert,
}

pub struct SequenceMatcher<'a> {
    a: &'a [String],
    b: &'a [String],
    b2j: HashMap<&'a str, Vec<usize>>,
    bpopular: HashSet<&'a str>,
}

impl<'a> SequenceMatcher<'a> {
    pub fn new(a: &'a [String], b: &'a [String], autojunk: bool) -> Self {
        // __chain_b
        let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, elt) in b.iter().enumerate() {
            b2j.entry(elt.as_str()).or_default().push(i);
        }
        let mut bpopular = HashSet::new();
        let n = b.len();
        if autojunk && n >= 200 {
            let ntest = n / 100 + 1;
            let popular: Vec<&str> = b2j.iter().filter(|(_, v)| v.len() > ntest).map(|(k, _)| *k).collect();
            for elt in popular {
                b2j.remove(elt);
                bpopular.insert(elt);
            }
        }
        SequenceMatcher { a, b, b2j, bpopular }
    }

    /// Longest matching block in a[alo:ahi] and b[blo:bhi]: (i, j, size). No junk (isjunk=None), so the junk
    /// extension loops of the Python version never run; popular elements are still not matched in the core loop
    /// but are picked up by the extension loops, as in Python (bjunk is empty, so only the "not popular" check).
    fn find_longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> (usize, usize, usize) {
        let (a, b) = (self.a, self.b);
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(a[i].as_str()) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        // Python: extend with non-junk elements first (isbjunk is always false here, so these are no-ops), then
        // with junk elements (none). Matching popular elements is what the next two loops would do with junk;
        // Python's loops test `not isbjunk(b[...])` / `isbjunk(b[...])`, where bjunk is empty: so the first pair
        // extends over equal elements on both sides, the second never does.
        while besti > alo && bestj > blo && !self.isbjunk(&b[bestj - 1]) && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi && bestj + bestsize < bhi && !self.isbjunk(&b[bestj + bestsize])
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    fn isbjunk(&self, _x: &str) -> bool {
        false
    }

    pub fn get_matching_blocks(&self) -> Vec<(usize, usize, usize)> {
        let (la, lb) = (self.a.len(), self.b.len());
        let mut queue = vec![(0, la, 0, lb)];
        let mut blocks = vec![];
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.find_longest_match(alo, ahi, blo, bhi);
            if k > 0 {
                blocks.push((i, j, k));
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        blocks.sort();
        // collapse adjacent blocks
        let (mut i1, mut j1, mut k1) = (0, 0, 0);
        let mut out = vec![];
        for (i2, j2, k2) in blocks {
            if i1 + k1 == i2 && j1 + k1 == j2 {
                k1 += k2;
            } else {
                if k1 > 0 {
                    out.push((i1, j1, k1));
                }
                (i1, j1, k1) = (i2, j2, k2);
            }
        }
        if k1 > 0 {
            out.push((i1, j1, k1));
        }
        out.push((la, lb, 0));
        let _ = &self.bpopular;
        out
    }

    pub fn get_opcodes(&self) -> Vec<(Tag, usize, usize, usize, usize)> {
        let (mut i, mut j) = (0, 0);
        let mut out = vec![];
        for (ai, bj, size) in self.get_matching_blocks() {
            let tag = if i < ai && j < bj {
                Some(Tag::Replace)
            } else if i < ai {
                Some(Tag::Delete)
            } else if j < bj {
                Some(Tag::Insert)
            } else {
                None
            };
            if let Some(t) = tag {
                out.push((t, i, ai, j, bj));
            }
            i = ai + size;
            j = bj + size;
            if size > 0 {
                out.push((Tag::Equal, ai, i, bj, j));
            }
        }
        out
    }

    pub fn ratio(&self) -> f64 {
        let matches: usize = self.get_matching_blocks().iter().map(|b| b.2).sum();
        let len = self.a.len() + self.b.len();
        if len == 0 {
            1.0
        } else {
            2.0 * matches as f64 / len as f64
        }
    }
}
