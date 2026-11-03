"""Writer and humanizer models, rebuilt from one base model plus a chain of LoRA adapters.

Every model was trained on top of the previous one, so each is the base Qwen2.5-1.5B-Instruct with the
adapters along its path merged in, in order. Merging the same way as PEFT's merge_and_unload (delta = B @ A
* alpha / r in float32 on the CPU, added to the bfloat16 weight) reproduces the trained models bit for bit.
"""
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = Path(__file__).resolve().parent / "models"
WRITER_CHAIN = ["sft", "r1", "r2", "r3"]
HUMANIZER_CHAIN = WRITER_CHAIN + ["humanizer", "rl"]
CHAINS = {"writer_r3": WRITER_CHAIN, "writer_sft": ["sft"],
          "humanizer_evo": HUMANIZER_CHAIN + ["evo1", "evo2", "evo3", "evo4"], "humanizer_rl": HUMANIZER_CHAIN}
LABELS = {"writer_r3": "Round 3 writer", "writer_sft": "SFT writer", "humanizer_evo": "Evolved humanizer",
          "humanizer_rl": "v2 RL humanizer"}

# ---- prompts, exactly as in training ----
DOMAIN_INSTRUCTIONS = {"review": "Write an online product review in your own voice.",
                       "creative": "Write a short piece of fiction.", "essay": "Write a student essay."}
WHAT = {"essay": "student essay", "abstract": "research abstract", "review": "product review", "creative": "story",
        "other": "text"}
HUMANIZE = ("Rewrite this passage from a {what} so it reads like a real person wrote it. Keep every point, fact "
            "and name. Write only the rewritten passage.\n\n{text}")


def write_prompt(domain: str, opening: str, topic: str | None = None) -> str:
    lines = [DOMAIN_INSTRUCTIONS[domain]]
    if topic:
        lines.append(f"Topic: {topic}")
    lines.append(f"It begins with this sentence, which is already written:\n{opening}")
    lines.append("Continue from right after that sentence. Write only the text itself.")
    return "\n\n".join(lines)


def chunks(text: str, max_sents: int = 4) -> list[list[str]]:
    """Paragraphs, each split into passages of at most max_sents sentences (the humanizer trained on 3-5)."""
    out = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        s = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", para) if x]
        n = max(1, round(len(s) / max_sents))
        size = -(-len(s) // n)
        out.append([" ".join(s[i:i + size]) for i in range(0, len(s), size)])
    return out


def load(name: str, device: str | None = None, progress=lambda msg: None):
    """Tokenizer and the merged model `name` (a key of CHAINS) on `device` (the GPU if there is one)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = MODELS / "base-1.5b"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device).eval()
    params = dict(model.named_parameters())
    chain = CHAINS[name]
    with torch.no_grad():
        for i, ad in enumerate(chain, 1):
            progress(f"merging {i}/{len(chain)}")
            scale = json.loads((MODELS / "adapters" / ad / "adapter.json").read_text())["scale"]
            w = load_file(MODELS / "adapters" / ad / "adapter.safetensors")
            for k, a in w.items():
                if ".lora_A." in k:
                    b = w[k.replace(".lora_A.", ".lora_B.")]
                    # delta on the CPU in float32 (as PEFT does), added on the GPU: identical result, much faster
                    params[k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")] += \
                        ((b @ a) * scale).to(device, non_blocking=True)
    return tok, model
