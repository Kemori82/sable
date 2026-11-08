//! Writer and humanizer models (each the base model plus a chain of LoRA adapters, in training order) and their
//! prompts, exactly as in training.

pub const WRITER_CHAIN: &[&str] = &["sft", "r1", "r2", "r3"];

pub fn chain(name: &str) -> Option<Vec<&'static str>> {
    let humanizer: Vec<&str> = [WRITER_CHAIN, &["humanizer", "rl"]].concat();
    Some(match name {
        "writer_r3" => WRITER_CHAIN.to_vec(),
        "writer_sft" => vec!["sft"],
        "humanizer_writer" => vec!["sft", "writer_voice"], // trained from the SFT writer on its own passages
        "humanizer_evo" => [humanizer.as_slice(), &["evo1", "evo2", "evo3", "evo4"]].concat(),
        "humanizer_rl" => humanizer,
        // Correct's meaning repair: Writer voice, then the repair model (supervised, then reinforcement round)
        "repair" => vec!["sft", "writer_voice", "repair_sft", "repair_rl"],
        _ => return None,
    })
}

pub fn label(name: &str) -> Option<&'static str> {
    Some(match name {
        "writer_r3" => "Round 3 writer",
        "writer_sft" => "SFT writer",
        "humanizer_writer" => "Writer-voice humanizer",
        "humanizer_evo" => "Evolved humanizer",
        "humanizer_rl" => "v2 RL humanizer",
        "humanizer_chain" => "Chain humanizer",
        "repair" => "repair model",
        _ => return None,
    })
}

/// The chain humanizer: each passage rewritten by these in turn, one try each.
pub const CHAIN_STEPS: [&str; 3] = ["humanizer_rl", "humanizer_writer", "humanizer_evo"];

pub fn what(kind: &str) -> &'static str {
    match kind {
        "essay" => "student essay",
        "abstract" => "research abstract",
        "review" => "product review",
        "creative" => "story",
        _ => "text",
    }
}

pub fn humanize_prompt(what: &str, text: &str) -> String {
    format!("Rewrite this passage from a {what} so it reads like a real person wrote it. Keep every point, fact \
             and name. Write only the rewritten passage.\n\n{text}")
}

/// The training prompt, plus an optional "What it should say" line. Without an opening sentence it writes the
/// whole text.
pub fn write_prompt(domain: &str, opening: &str, topic: Option<&str>, instructions: Option<&str>) -> String {
    let mut lines = vec![match domain {
        "review" => "Write an online product review in your own voice.",
        "creative" => "Write a short piece of fiction.",
        _ => "Write a student essay.",
    }
    .to_string()];
    if let Some(t) = topic {
        lines.push(format!("Topic: {t}"));
    }
    if let Some(i) = instructions {
        lines.push(format!("What it should say: {i}"));
    }
    if !opening.is_empty() {
        lines.push(format!("It begins with this sentence, which is already written:\n{opening}"));
        lines.push("Continue from right after that sentence. Write only the text itself.".into());
    } else {
        lines.push("Write only the text itself.".into());
    }
    lines.join("\n\n")
}

/// The repair model's prompt (as trained, aimodel/repair/teacher_fixes.py PROMPT): fix the sentence in [[ ]] of the
/// rewrite so it says what the original sentence says.
pub fn repair_prompt(orig_passage: &str, original: &str, rewrite_marked: &str) -> String {
    format!("A rewrite of a passage lost or changed some of the original's meaning. Fix one sentence of the rewrite.

ORIGINAL PASSAGE: {orig_passage}

ORIGINAL SENTENCE (what the fixed sentence must say): {original}

REWRITE (the sentence to fix is in [[double brackets]]): {rewrite_marked}

Write only the fixed sentence. It must say everything the original sentence says, with every fact, name, number and
detail right, and nothing it doesn't say. Keep the rewrite's plain, casual wording and its style: change as few
words as you can, and use a simple word where the original uses a fancy one.")
}
