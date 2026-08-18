import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mech_interp_utils import MODEL_NAME, answer_token_id, balanced_subset, model_device_inputs, model_load_kwargs, role_margin


def run_head_ablation(sample_size=160):
    """Ablate heads selected by the current descriptive analysis, not a stale list."""
    candidates_path = ROOT / "results" / "attention_head_candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError("Run attention/head_analysis.py first to create head candidates.")
    candidates = json.loads(candidates_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_load_kwargs())
    model.eval()
    with open(ROOT / "data" / "aligned_dataset.json") as f:
        dataset = json.load(f)
    dataset = balanced_subset(dataset, sample_size)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    def score():
        margins = []
        for item in dataset:
            inputs = model_device_inputs(tokenizer, item["prompt"])
            correct = answer_token_id(tokenizer, item["prompt"], item["agent"])
            incorrect = answer_token_id(tokenizer, item["prompt"], item["patient"])
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
            margins.append(role_margin(logits, correct, incorrect).item())
        return sum(margins) / len(margins)

    baseline = score()
    results = {}
    for candidate in candidates:
        layer, head = candidate["layer"], candidate["head"]
        def ablate(module, args, head=head):
            states = args[0].clone()
            states[..., head * head_dim:(head + 1) * head_dim] = 0
            return (states, *args[1:])
        handle = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(ablate)
        try:
            results[f"L{layer}.H{head}"] = baseline - score()
        finally:
            handle.remove()

    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("Drop in mean correct-vs-incorrect logit margin")
    plt.title("Single-head ablation impact")
    plt.savefig(ROOT / "results" / "head_ablation_impact.png", dpi=300, bbox_inches="tight")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_head_ablation()
